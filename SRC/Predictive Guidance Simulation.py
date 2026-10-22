import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# Camera lock to pan tilt only

try:
    mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'
except KeyError:
    pass

#flight path variables


Straight_time = 20
curve_time = 5
Straight_time2 = 20
targ_vel = 1000 #velocity of the target aircraft in m/s
turn_angle = np.pi/2

tmax = 45 #time after which the simulation stops, seconds
playback_time = 25 #for speeding up or slowing down the animation, seconds
dt = 0.1 #time step for the simulation, seconds
yz_angle = 0.5#angle in radians at which the aircraft turns in the local YZ plane, i.e. roll before the turn begins. 0 = level turn, pi/2 = vertical turn, -pi/2 = inverted vertical turn
aircraft_start_loc = np.array([-35000, -20000,21000])#xyz coordinates of the aircraft start location (z is altitude)
Steps = 20000 #animation steps

points_per_step = 9 #used instead of a lower dt, total points ever detected = points_per_step * tmax/dt, so 45*9/0.1 = 4050 points
noise_sd = 10.0 #standard deviation of the spatial noise in meters, applied to each of x,y,z independently
time_noise_sd = 0.01 #standard deviation of the temporal noise in seconds, applied to the time of each point independently. 0 = no temporal noise, 0.1 = 100ms std dev, etc.
recent_window = 1 #seconds, how far back in time to consider points for the local linear fit. 1 = 1 second of points, 0.5 = 0.5 seconds of points, etc.
projection_time = 15   # seconds ahead to illustrate the projected position of the aircraft based on the local linear fit. 15 = 15 seconds ahead, etc. for animation only, does not affect the missile intercept calculations

missile_speed = 1300                        # m/s, constant speed of missile
missile_start = np.array([0.0, 0.0, 0.0])   # launch point
hit_radius = 10.0                           # m, counts as an intercept
missile_launch_delay = 3.0                  # s, missile launches at this sim time

# ============================================================
# =============== INTERNAL STATE VARIABLES ====================
# ============================================================

curve_start_x = None
curve_start_y = None
curve_start_z = None
curve_initialized = False

straight2_start_x = None
straight2_start_y = None
straight2_start_z = None
straight2_initialized = False

radius = (targ_vel * curve_time) / turn_angle

center_x = None
center_y = None
center_z = None

# ============================================================
# ================== TRAJECTORY FUNCTION =====================
# ============================================================

def target_location(t, target_states):
    """
    Compute the target aircraft's position at time t, given the
    history of target_states (for determining the start of the curve).
    Always assume starting in the +X direction, then turning in the YZ plane at the specified angle.

    0 <= t <= Straight_time: straight line

    Straight_time < t <= Straight_time + curve_time: curved turn

    Straight_time + curve_time < t <= Straight_time + curve_time + Straight_time2: straight line after turn

    t > Straight_time + curve_time + Straight_time2: hold last position

    returns a numpy array of shape (3,) representing the true (x, y, z) position of the target aircraft at time t
    This is the data generation function, not the noisy measurements, which are generated later.
    """

    global curve_start_x, curve_start_y, curve_start_z
    global curve_initialized

    global straight2_start_x, straight2_start_y, straight2_start_z
    global straight2_initialized

    global center_x, center_y, center_z

    if 0 <= t <= Straight_time:

        x = aircraft_start_loc[0] + targ_vel * t
        y = aircraft_start_loc[1]
        z = aircraft_start_loc[2]

    elif Straight_time < t <= Straight_time + curve_time:

        tc = t - Straight_time

        if not curve_initialized:

            if len(target_states) > 0:
                curve_start_x = target_states[-1, 0]
                curve_start_y = target_states[-1, 1]
                curve_start_z = target_states[-1, 2]

            else:
                curve_start_x = (
                    aircraft_start_loc[0]
                    + targ_vel * Straight_time
                )
                curve_start_y = aircraft_start_loc[1]
                curve_start_z = aircraft_start_loc[2]

            center_x = curve_start_x
            center_y = (
                curve_start_y
                + radius * np.cos(yz_angle)
            )
            center_z = (
                curve_start_z
                + radius * np.sin(yz_angle)
            )

            curve_initialized = True

        angle = tc * turn_angle / curve_time

        arc_angle = -np.pi / 2 + angle

        x = center_x + radius * np.cos(arc_angle)

        y = (
            center_y
            + radius
            * np.sin(arc_angle)
            * np.cos(yz_angle)
        )

        z = (
            center_z
            + radius
            * np.sin(arc_angle)
            * np.sin(yz_angle)
        )

    elif (
        Straight_time + curve_time
        < t
        <= Straight_time + curve_time + Straight_time2
    ):

        if not straight2_initialized:

            straight2_start_x = target_states[-1, 0]
            straight2_start_y = target_states[-1, 1]
            straight2_start_z = target_states[-1, 2]

            straight2_initialized = True

        ts = t - (Straight_time + curve_time)

        dx = np.cos(turn_angle)
        dy = np.sin(turn_angle) * np.cos(yz_angle)
        dz = np.sin(turn_angle) * np.sin(yz_angle)

        x = straight2_start_x + targ_vel * ts * dx
        y = straight2_start_y + targ_vel * ts * dy
        z = straight2_start_z + targ_vel * ts * dz

    else:

        if straight2_initialized:

            ts_max = Straight_time2

            dx = np.cos(turn_angle)
            dy = np.sin(turn_angle) * np.cos(yz_angle)
            dz = np.sin(turn_angle) * np.sin(yz_angle)

            x = (
                straight2_start_x
                + targ_vel * ts_max * dx
            )

            y = (
                straight2_start_y
                + targ_vel * ts_max * dy
            )

            z = (
                straight2_start_z
                + targ_vel * ts_max * dz
            )

        else:

            x = (
                aircraft_start_loc[0]
                + targ_vel * Straight_time
            )

            y = aircraft_start_loc[1]
            z = aircraft_start_loc[2]

    return np.array([x, y, z])


# ============================================================
# ================== GENERATE TRAJECTORY ======================
# ============================================================

curve_initialized = False
straight2_initialized = False

times = np.arange(0, tmax, dt)
n_points = len(times)

target_states = np.zeros((n_points, 3))

for i in range(n_points):
    target_states[i] = target_location(
        times[i],
        target_states[:i]
    )

# Ground-truth velocity for comparison (finite difference of the clean path)
true_vel = np.gradient(target_states, dt, axis=0)


# ============================================================
# ===================== PLOTTING SETUP ========================
# ============================================================

fig = plt.figure(figsize=(14, 10))

ax = fig.add_subplot(111, projection='3d')

all_points = np.vstack([target_states, missile_start])

padding = 0.1 #amount of extra space around the trajectory in the plot, as a fraction of the max range. currently +10% of the max range in each dimension

x_range = np.ptp(all_points[:, 0])
y_range = np.ptp(all_points[:, 1])
z_range = np.ptp(all_points[:, 2])

max_range = max(
    x_range,
    y_range,
    z_range
)
#where to center the plot in each dimension, based on the min and max of all points
x_center = (
    np.max(all_points[:, 0])
    + np.min(all_points[:, 0])
) / 2

y_center = (
    np.max(all_points[:, 1])
    + np.min(all_points[:, 1])
) / 2

z_center = (
    np.max(all_points[:, 2])
    + np.min(all_points[:, 2])
) / 2

plot_radius = max_range / 2 * (1 + padding)
#set the limits of the plot in each dimension, centered around the trajectory with some padding
ax.set_xlim(
    x_center - plot_radius,
    x_center + plot_radius
)

ax.set_ylim(
    y_center - plot_radius,
    y_center + plot_radius
)

ax.set_zlim(
    z_center - plot_radius,
    z_center + plot_radius
)

ax.set_box_aspect([1, 1, 1])

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.set_title(
    '3D Trajectory with Historical + Recent '
    'Gaussian Clouds (X, Y, Z, T Noise)'
)

ax.grid(True)


# ============================================================
# ===================== CAMERA DEFAULTS =======================
# ============================================================

# Starting view only; after this the mouse controls the camera
ax.view_init(elev=20, azim=45, roll=0)


# ============================================================
# ====================== PLOT OBJECTS =========================
# ============================================================
#initialize empty plot objects for the target, historical cloud, recent cloud, projection line, missile, missile trail, and aim point
#does not plot anything yet, just sets up the objects to be updated in the animation
target_point, = ax.plot(
    [], [], [],
    'ro',
    markersize=8
)

hist_scatter = ax.scatter(
    [], [], [],
    s=3,
    c='blue',
    alpha=0.35
)

recent_scatter = ax.scatter(
    [], [], [],
    s=6,
    c='green',
    alpha=0.8
)

# Projection line: fitted position now -> projected position
proj_line, = ax.plot(
    [], [], [],
    'r--',
    linewidth=1.5
)

# Missile, its trail, and the intercept point it is aiming at
missile_point, = ax.plot(
    [], [], [],
    'ks',
    markersize=7
)

missile_trail, = ax.plot(
    [], [], [],
    'k-',
    linewidth=1.2
)

aim_point, = ax.plot(
    [], [], [],
    'mx',
    markersize=10,
    markeredgewidth=2
)

historical_cloud = []#list of all generated scattered points, used for plotting the historical point cloud
recent_cloud = []#list of all generated scattered points within the recent_window, used for plotting the recent point cloud and for the local linear fit

# Store every generated scattered measurement:
# [actual_time, noisy_time, x, y, z]
saved_points = []

# Velocity estimates from the local linear fit:
# [t_now, vx_est, vy_est, vz_est, vx_true, vy_true, vz_true]
velocity_log = []

# Missile state initialisation
missile_state = {
    "pos": missile_start.astype(float).copy(),
    "trail": [missile_start.astype(float).copy()],
    "last_t": None,
    "hit": False,
    "hit_time": None,
    "min_dist": np.inf,
}


# ============================================================
# ===================== ANIMATION FUNCS =======================
# ============================================================

def point_segment_distance(p, a, b):
    """
    input: p = True point(x,y,z) of target, a = segment start, b = segment end

    Compute the shortest distance from the missile to the path between 2 missile locations, a and b,
    i.e. how close the missile came to the aircraft during this time step
    and return the closest point on the segment to p.

    Assume detonation is based on true proximity, not the noisy measurements, so this is used to determine if the missile hit the aircraft.
    """
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return np.linalg.norm(p - a), a.copy()
    u = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    closest = a + u * ab
    return np.linalg.norm(p - closest), closest


def init():
    """
    Initialize the animation by clearing all plot objects.
    Does not initialise the state variables, which are already initialized above. This is only for the animation.
    """

    target_point.set_data([], [])
    target_point.set_3d_properties([])

    proj_line.set_data([], [])
    proj_line.set_3d_properties([])

    missile_point.set_data([missile_start[0]], [missile_start[1]])
    missile_point.set_3d_properties([missile_start[2]])

    missile_trail.set_data([], [])
    missile_trail.set_3d_properties([])

    aim_point.set_data([], [])
    aim_point.set_3d_properties([])

    hist_scatter._offsets3d = (
        [],
        [],
        []
    )

    recent_scatter._offsets3d = (
        [],
        [],
        []
    )

    return (
        target_point,
        hist_scatter,
        recent_scatter,
        proj_line,
        missile_point,
        missile_trail,
        aim_point
    )


def update(frame):
    """
    Update the animation for the given frame index.
    This function is called by FuncAnimation for each frame of the animation.
    It updates the target position, generates noisy measurements, updates the historical and recent clouds,
    performs the local linear fit to estimate velocity, updates the missile position and trail, and updates the plot objects accordingly.
    """
    t_now = times[frame]

    px, py, pz = target_states[frame]


  
    # Moves aircraft dot to position this frame
    

    target_point.set_data(
        [px],
        [py]
    )

    target_point.set_3d_properties(
        [pz]
    )


    # NOISY SAMPLE GENERATION
    #uses true target position at current time, adds temporal noise and spatial noise 
    #generates a cloud of points around the true position, simulating noisy measurements from a sensor

    noisy_pts_list = []

    for _ in range(points_per_step):

        # Temporal noise
        t_noisy = max(
            0.0,
            np.random.normal(
                t_now,
                time_noise_sd
            )
        )

        # Position corresponding to noisy time
        base_pos_noisy_t = target_location(
            t_noisy,
            target_states
        )

        # Spatial noise
        spatial_noise = np.random.normal(
            0.0,
            noise_sd,
            size=3    #has to do for x,y,z independently
        )

        final_noisy_pt = (
            base_pos_noisy_t
            + spatial_noise #position uncertainty implemented here
        )

        noisy_pts_list.append(
            final_noisy_pt
        )

        # Save every generated scattered point
        saved_points.append([
            t_now,
            t_noisy,
            final_noisy_pt[0],
            final_noisy_pt[1],
            final_noisy_pt[2]
        ])


    noisy_pts = np.array(noisy_pts_list)


    # ========================================================
    # HISTORICAL CLOUD
    # ========================================================

    historical_cloud.append(
        noisy_pts
    )


    # ========================================================
    # RECENT CLOUD
    # ========================================================

    recent_cloud.append(
        (
            noisy_pts,
            t_now
        )
    )

    cutoff = t_now - recent_window

    recent_cloud[:] = [
        (pts, ts)
        for pts, ts in recent_cloud
        if ts >= cutoff
    ]


    # ========================================================
    # FLATTEN HISTORICAL CLOUD
    # ========================================================

    hist_pts = np.vstack(
        historical_cloud
    )


    # ========================================================
    # FLATTEN RECENT CLOUD
    # ========================================================

    if len(recent_cloud) > 0:

        recent_pts = np.vstack([
            pts
            for pts, ts in recent_cloud
        ])

    else:

        recent_pts = np.zeros(
            (0, 3)
        )


    # LOCAL LINEAR FIT -> VELOCITY VECTOR + PROJECTION
  

    if len(recent_cloud) > 0:
        recent_t = np.concatenate([
            np.full(len(pts), ts) for pts, ts in recent_cloud
        ])
    else:
        recent_t = np.array([])

    # Need at least 2 distinct timestamps for a slope
    if len(np.unique(recent_t)) >= 2:

        # Fit against time relative to "now", so the intercept is
        # the fitted position at t_now and the slope is the velocity.
        coeffs = np.polyfit(recent_t - t_now, recent_pts, deg=1)

        vel_est = coeffs[0]       # (vx, vy, vz)
        pos_now = coeffs[1]       # fitted position at t_now

        vx_est, vy_est, vz_est = vel_est
        vx_true, vy_true, vz_true = true_vel[frame]

        velocity_log.append([
            t_now,
            vx_est, vy_est, vz_est,
            vx_true, vy_true, vz_true
        ])

        # Projection: fitted position now -> position projection_time ahead
        pos_future = pos_now + vel_est * projection_time
        #projection line to illustrate velocity for animation only, does not affect the missile intercept calculations
        proj_line.set_data(
            [pos_now[0], pos_future[0]],
            [pos_now[1], pos_future[1]]
        )

        proj_line.set_3d_properties(
            [pos_now[2], pos_future[2]]
        )


        # ----------------------------------------------------
        # MISSILE: constant speed, aimed at the predicted
        # intercept point (recomputed every frame)
        # ----------------------------------------------------

        if not missile_state["hit"] and t_now >= missile_launch_delay: #if the chase has started and the missile has not hit yet

            m_old = missile_state["pos"].copy()

            # in tau seconds, the plane will be assumed to be at pos_now + vel_est * tau
            # the missile will be at its current position + missile_speed * tau * heading_unit_vector (direction of the missile)
            #unit vector has len =1, |displacement + v Tau| = |missile_speed * tau|
            #square both sides to get a quadratic equation in tau, then solve for tau
            d = pos_now - m_old #current distance vector from missile to target
            a_q = np.dot(vel_est, vel_est) - missile_speed**2 #
            b_q = 2.0 * np.dot(d, vel_est)
            c_q = np.dot(d, d)

            tau = None
#solving the quadratic equation for tau, which represents the time until intercept
            if abs(a_q) < 1e-9:
                if abs(b_q) > 1e-9 and -c_q / b_q > 0:
                    tau = -c_q / b_q
            else:
                disc = b_q**2 - 4.0 * a_q * c_q
                if disc >= 0:
                    roots = [
                        (-b_q - np.sqrt(disc)) / (2.0 * a_q),
                        (-b_q + np.sqrt(disc)) / (2.0 * a_q)
                    ]
                    roots = [r for r in roots if r > 0]
                    if roots:
                        tau = min(roots)

            if tau is not None:
                aim = pos_now + vel_est * tau #aim point is the predicted position of the target at the time of intercept, which the missile is guided towards
            else:
                # No intercept solution: aim at the current fitted position
                aim = pos_now.copy()

            # Time since the last missile update (0 on the first estimate,
            # so launch happens as soon as a velocity estimate exists)
            if missile_state["last_t"] is None:
                step_dt = 0.0
            else:
                step_dt = t_now - missile_state["last_t"]

            missile_state["last_t"] = t_now

            heading = aim - m_old #vector from missile to aim point, which is the direction the missile should move in this time step
            heading_norm = np.linalg.norm(heading)

            if heading_norm > 1e-9: #prevent division by zero, if the missile is already at the aim point, it does not move
                m_new = m_old + heading / heading_norm * missile_speed * step_dt #missile steps toward aim point at constant speed, distance scaled by the time step
            else:
                m_new = m_old

            # Hit check against the TRUE aircraft position, along this step
            miss, closest = point_segment_distance(
                target_states[frame], m_old, m_new
            )

            missile_state["min_dist"] = min(missile_state["min_dist"], miss)

            if miss <= hit_radius: #if its within the hit radius, the missile has hit the target
                missile_state["hit"] = True
                missile_state["hit_time"] = t_now
                m_new = closest

            missile_state["pos"] = m_new #update the missile position to the new position after this time step
            missile_state["trail"].append(m_new.copy()) #update the missile trail with the new position it has moved to

            trail = np.array(missile_state["trail"])

            missile_point.set_data([m_new[0]], [m_new[1]])
            missile_point.set_3d_properties([m_new[2]])

            missile_trail.set_data(trail[:, 0], trail[:, 1])
            missile_trail.set_3d_properties(trail[:, 2])

            if missile_state["hit"]:
                aim_point.set_data([], [])
                aim_point.set_3d_properties([])
            else:
                aim_point.set_data([aim[0]], [aim[1]])
                aim_point.set_3d_properties([aim[2]])

        if missile_state["hit"]: #check if the missile has hit the target, if so display the hit time in the title
            status = f"HIT at t={missile_state['hit_time']:.1f}s"
        elif t_now < missile_launch_delay: #check if the missile has not launched yet, if so display the time until launch in the title
            status = f"launch in {missile_launch_delay - t_now:.1f}s"
        else:#display the current range to the target in the title
            rng = np.linalg.norm(target_states[frame] - missile_state["pos"])
            status = f"range {rng:.0f} m"

        ax.set_title(
            f"t={t_now:.1f}s  "
            f"v_est=({vx_est:.0f}, {vy_est:.0f}, {vz_est:.0f})  "
            f"v_true=({vx_true:.0f}, {vy_true:.0f}, {vz_true:.0f}) m/s\n"
            f"missile: {status}"
        )


    # ========================================================
    # UPDATE SCATTER OBJECTS
    # ========================================================

    hist_scatter._offsets3d = (
        hist_pts[:, 0],
        hist_pts[:, 1],
        hist_pts[:, 2]
    )#display the historical cloud of points in blue, with low opacity

    recent_scatter._offsets3d = (
        recent_pts[:, 0],
        recent_pts[:, 1],
        recent_pts[:, 2]
    )#display the recent cloud of points in green, with higher opacity


    return (
        target_point,
        hist_scatter,
        recent_scatter,
        proj_line,
        missile_point,
        missile_trail,
        aim_point
    )

#animation speed control
frame_skip = max(1, len(times) // Steps)
frames = range(0, len(times), frame_skip)

interval_ms = 1000 * playback_time / len(frames)#animation interval in milliseconds, scaled to the playback time and number of frames

anim = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init,
    blit=False,
    interval=interval_ms,
    repeat=False
)
plt.show()
# ============================================================
# ======================= SAVE CSV ============================
# ============================================================

saved_points = np.array(saved_points)

np.savetxt(
    "scattered_points.csv",
    saved_points,
    delimiter=",",
    header="t_actual,t_noisy,x,y,z",
    comments="",
    fmt="%.6f"
)

print(
    f"Saved {len(saved_points)} scattered points "
    f"to scattered_points.csv"
)

velocity_log = np.array(velocity_log)

np.savetxt(
    "velocity_estimates.csv",
    velocity_log,
    delimiter=",",
    header="t,vx_est,vy_est,vz_est,vx_true,vy_true,vz_true",
    comments="",
    fmt="%.6f"
)

print(
    f"Saved {len(velocity_log)} velocity estimates "
    f"to velocity_estimates.csv"
)

if missile_state["hit"]:
    print(
        f"Missile intercepted the aircraft at "
        f"t={missile_state['hit_time']:.1f} s"
    )
else:
    print(
        f"Missile did not intercept; closest approach "
        f"{missile_state['min_dist']:.0f} m"
    )
