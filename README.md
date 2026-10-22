# 3D Aircraft Trajectory Estimation and Interception Simulation

**MMA3001 – Numerical Methods and Machine Learning, Individual Project (2026)**

## Project overview

This project investigates how numerical methods can estimate an aircraft's motion from noisy position measurements and use those estimates to predict its future trajectory. A simulated aircraft follows a three-dimensional flight path while a measurement system collects position observations containing spatial and timing noise.

The project uses local linear least-squares fitting to estimate the aircraft's velocity from recent observations. The estimated velocity is then used to predict future position and calculate a potential interception point for a simulated constant-speed interceptor. The simulation visualises the aircraft trajectory, noisy measurements, velocity estimates and interception behaviour over time.

**In brief.** The project explores the effect of measurement noise and finite observation windows on velocity estimation and future-position prediction. Because the true simulated trajectory is known, estimated positions and velocities can be compared against ground truth to evaluate numerical accuracy. The simulation also provides a framework for investigating how estimation errors affect predicted interception outcomes.

## Contents

1. Engineering problem
2. Inputs and outputs
3. Computational approach
4. Validation and results
5. Sensitivity analysis
6. Limitations
7. Getting started
8. Repository layout
9. Testing, documentation and reproducibility
10. AI use
11. Licence

## 1. Engineering problem

### Context

A moving aircraft's future position can be estimated from observations of its past motion. In practical applications, measurements are imperfect: position estimates contain noise, timestamps may be uncertain, and velocity must often be inferred rather than measured directly.

These uncertainties affect the accuracy of trajectory predictions. Errors in estimated velocity can grow into substantial position errors when predictions are made further into the future.

### Problem

The project investigates the following problem:

Given a sequence of noisy three-dimensional position measurements with uncertain timestamps, estimate the aircraft's velocity and predict its future position.

The resulting prediction is then used in a simplified interception simulation to investigate how trajectory-estimation errors affect the predicted interception point.

The principal objectives are to:

- Generate a known three-dimensional aircraft trajectory.
- Simulate noisy position observations and timing uncertainty.
- Estimate velocity using local linear least-squares fitting.
- Predict future aircraft positions from the estimated motion.
- Compare numerical estimates with the known simulated trajectory.
- Investigate how measurement noise and estimation settings affect prediction accuracy.

### Intended use

The simulation is intended as an engineering investigation into numerical estimation and prediction under measurement uncertainty. It provides a controlled environment in which estimation methods can be evaluated against known reference values.

### Scope

The project uses a simulated aircraft trajectory rather than real flight data. The aircraft motion and interceptor behaviour are simplified mathematical models. Results therefore demonstrate the behaviour of the implemented methods under the chosen simulation assumptions; they do not establish performance in a real aircraft tracking or interception system.

## 2. Inputs and outputs

### Inputs

The simulation uses configurable parameters, including:

| Input | Description | Units |
|---|---|---|
| Trajectory duration | Duration of the simulated flight | seconds |
| Target speed | Speed used to generate the aircraft trajectory | m/s |
| Initial aircraft position | Starting three-dimensional position | metres |
| Spatial noise standard deviation | Standard deviation of simulated position noise | metres |
| Timing noise standard deviation | Standard deviation of simulated timestamp noise | seconds |
| Measurement frequency | Rate at which observations are generated | Hz or observations per second |
| Estimation window | Recent measurements used to estimate velocity | seconds |
| Interceptor speed | Constant speed used by the simplified interceptor model | m/s |
| Interceptor launch delay | Time between simulation start and interceptor launch | seconds |

The exact values are defined in the simulation's configuration section.

### Outputs

The simulation produces:

- A three-dimensional visualisation of the true aircraft trajectory.
- Noisy position measurements.
- Estimated aircraft velocity vectors.
- Predicted future positions and interception points.
- A simulated interceptor trajectory.
- CSV files containing measurement data and velocity estimates, where enabled in the script.

The measurement data include the simulated observation time, noisy timestamp and measured spatial coordinates. The velocity-estimation output records the estimated motion at the relevant simulation times.

## 3. Computational approach

The computational workflow is:

```text
Generate true aircraft trajectory
              |
              v
Generate noisy position observations
              |
              v
Collect recent measurements
              |
              v
Fit local linear least-squares model
              |
              v
Estimate velocity and current position
              |
              v
Predict future aircraft position
              |
              v
Calculate simplified interception point
              |
              v
Animate results and export data
```

### 3.1 Trajectory generation

A synthetic aircraft trajectory is generated from prescribed flight segments, including straight flight and a curved manoeuvre. The simulation retains the true trajectory so that numerical estimates can be compared with reference positions and velocities.

### 3.2 Measurement noise

Gaussian noise is added independently to the three spatial coordinates. Timing uncertainty is simulated by perturbing the observation timestamps.

This creates a controlled dataset in which the effects of measurement uncertainty can be investigated by varying the noise parameters.

### 3.3 Velocity estimation

Recent observations are used to fit a local linear model to position as a function of time. The fitted slope provides an estimate of the three-dimensional velocity vector.

The method is computationally straightforward and provides a local approximation of the aircraft's motion. Its accuracy depends on factors including the observation window, measurement noise, sampling frequency and curvature of the underlying trajectory.

### 3.4 Future-position prediction

The estimated position and velocity are used to predict the aircraft's future location under a constant-velocity approximation. The prediction becomes less representative of the true motion when the aircraft accelerates or changes direction significantly.

### 3.5 Interception simulation

The predicted target motion is used to calculate a potential interception point for a simulated interceptor moving at constant speed. The animation then compares the interceptor's motion with the target's trajectory.

This component is a simplified mathematical demonstration, not a validated physical interception model.

## 4. Validation and results

The known simulated trajectory provides ground truth for evaluating the numerical methods.

The primary validation should compare the estimated velocity and predicted position with their corresponding true values at the same times.

Recommended evaluation metrics include:

- **Velocity error:** Euclidean difference between the estimated and true velocity vectors, in m/s.
- **Position prediction error:** Euclidean distance between the predicted and true positions, in metres.
- **Root mean squared error (RMSE):** A summary measure that penalises larger estimation errors.
- **Runtime:** Time required to process observations and produce estimates.

A baseline method should also be evaluated, such as finite-difference velocity estimation from consecutive observations. Comparing this baseline with local linear least-squares fitting helps determine whether the additional fitting procedure improves accuracy.

Results should be reported for multiple noise levels and, where appropriate, different estimation-window durations. Quantitative conclusions should be added after these experiments have been run.

## 5. Sensitivity analysis

The simulation allows key parameters to be varied systematically.

Potential investigations include:

- Increasing spatial measurement noise.
- Increasing timestamp uncertainty.
- Changing the number of observations used in each fit.
- Changing the sampling frequency.
- Comparing straight and curved portions of the trajectory.
- Investigating how prediction error changes with forecast horizon.

The purpose is to determine which factors have the greatest effect on numerical accuracy and to identify trade-offs between noise reduction, responsiveness to manoeuvres and computational cost.

## 6. Limitations

The main limitations are:

- **Synthetic data:** Results may not generalise to real sensor measurements.
- **Simplified aircraft dynamics:** The trajectory is prescribed rather than generated by a complete flight-dynamics model.
- **Measurement assumptions:** Gaussian noise is an assumed model and may not represent every source of real measurement error.
- **Local linear approximation:** Velocity estimates and future-position predictions can become inaccurate during rapid manoeuvres.
- **Timing uncertainty:** Incorrect or inconsistent timestamps can affect the fitted velocity.
- **Simplified interceptor model:** The interception calculation does not represent a complete guidance, control or aerodynamic model.
- **Numerical validation requirements:** A visually plausible animation is not sufficient to establish that the estimates are accurate.

These limitations should be considered when interpreting the results.

## 7. Getting started

### Requirements

The project uses Python and scientific-computing libraries. The expected dependencies include:

- Python
- NumPy
- Matplotlib

Additional dependencies should be listed in the project's requirements file if used by the final implementation.

### Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project's dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the simulation:

```bash
python <your-main-script>.py
```

Replace the placeholder filename with the actual name of the main Python script.

### Output files

The simulation can export measurement observations and velocity estimates to CSV for further analysis. Generated files should be saved in a documented output directory rather than mixed with source code.

## 8. Repository layout

The repository is intended to contain the following structure:

```text
aircraft-trajectory-project/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── src/
│   └── trajectory_project/
│       ├── trajectory.py
│       ├── measurements.py
│       ├── estimation.py
│       └── prediction.py
├── tests/
│   ├── test_trajectory.py
│   ├── test_measurements.py
│   └── test_estimation.py
├── scripts/
│   └── run_simulation.py
├── results/
│   ├── plots/
│   └── tables/
├── docs/
└── test_report.md
```

This is a proposed structure. Files should only be listed as completed project components once they have actually been created.

## 9. Testing, documentation and reproducibility

### Testing

Automated tests should verify that:

- Trajectory generation returns finite three-dimensional positions.
- Measurement generation produces the expected coordinate shapes.
- Noise-free observations reproduce the expected reference trajectory.
- Velocity estimation returns finite values with the correct dimensions.
- Estimation error behaves as expected under controlled noise-free and noisy conditions.
- Prediction functions handle invalid inputs and insufficient observations appropriately.
- CSV exports contain the expected columns and valid data.

Tests can be run using `pytest`:

```bash
python -m pip install pytest
pytest -v
```

The actual test results and any known failures should be recorded in `test_report.md`.

### Documentation

Public functions should include docstrings describing their purpose, parameters, units, return values and relevant assumptions. HTML documentation can be generated using a tool such as `pdoc` once the source code has been organised into importable modules.

### Reproducibility

The repository should include dependency information, documented run instructions and the configuration required to reproduce the principal experiments. Random seeds should be configurable where repeatable noise realisations are required.

The final results should be accompanied by the parameter settings used to produce them.

## 10. AI use

Any use of AI tools during development should be disclosed in accordance with the MMA3001 project requirements.

The reflection should identify which tools were used, the tasks they assisted with, how suggestions and generated code were checked, any errors identified, and which technical decisions remained the student's responsibility.

The final submission should reflect the actual development process and the student's own understanding of the implemented numerical methods.

## 11. Licence

The repository should include a `LICENSE` file specifying the terms under which the original project code may be used and redistributed.

Any third-party code, datasets or other resources must be handled according to their respective licences and permissions.
