# Predictive Payload-Aware Obstacle Avoidance in PyBullet

This project implements and evaluates a predictive safety-control framework for a Franka Panda pick-and-place task in PyBullet. It combines a lightweight constant-velocity Kalman filter, an Artificial Potential Field (APF) avoidance controller, dual RGB-D-style perception, closest-point AABB obstacle handling, and payload-aware clearance checking.

The project is best treated as applied software-engineering research: the code implements the robotic control system, and the submitted CSV/figure outputs provide the recorded benchmark evidence used in the final report.

## Repository Structure

```text
.
|-- main.py                         # interactive PyBullet demo
|-- environmen.py                   # simulation setup
|-- obstacle.py                     # dynamic obstacle model
|-- predictor.py                    # Kalman filter predictor
|-- avoidance.py                    # APF and avoidance logic
|-- control.py                      # robot control and task execution
|-- tests/
|   |-- benchmark_raw_zero.py       # velocity benchmark, strict zero-collision criterion
|   |-- benchmark_runner.py         # narrow-passage benchmark
|   |-- benchmark_ablation.py       # proactive vs reactive ablation
|   |-- benchmark_visual_demo.py    # GUI demo for manual Figure 6.1 screenshot
|   |-- analyze_zero_collision.py   # success-rate summary from recorded CSV
|   |-- test_robustness.py          # noisy-observation KF check used for Figure 6.2.3
|   |-- plot_results.py             # Figures 6.2.1 and 6.2.2
|   |-- plot_ablation.py            # Figure 6.2.6
|   |-- plot_section_6_2_4_generalisation.py
|   `-- result/                     # final CSV/PNG outputs used in the report
`-- 

The final report results are based on the files in `tests/result/`.

## Requirements

Use Python 3.10 or later. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The main external packages are `pybullet`, `numpy`, `pandas`, and `matplotlib`.

## Running the Demo

From the project root:

```bash
python main.py
```

This opens the PyBullet simulation and runs the pick-and-place task with dynamic obstacle avoidance.

## Reproducing the Report Figures

The final report uses the recorded benchmark outputs in `tests/result/`. To regenerate the CSV-derived figures without re-running the simulations:

```bash
python tests/analyze_zero_collision.py
python tests/test_robustness.py
python tests/plot_results.py
python tests/plot_ablation.py
python tests/plot_section_6_2_4_generalisation.py
```

Generated CSV and PNG files are written to `tests/result/`.

`Figure 6.1` comes from a manual screenshot of the PyBullet scene shown by `tests/benchmark_visual_demo.py`.

## Re-running the Benchmarks

The full benchmarks are computationally heavier than figure generation:

```bash
python tests/benchmark_raw_zero.py
python tests/benchmark_runner.py
python tests/benchmark_ablation.py
```

These scripts write their outputs to `tests/result/`.

## Reproducibility Note

The benchmark uses stochastic obstacle motion, noisy observations, multiprocessing, and small initial-condition perturbations. Therefore, re-running the full Monte Carlo benchmark may produce slightly different trial-level CSV values unless the same random seeds, package versions, CPU scheduling, and simulation settings are reproduced exactly.

The CSV files in `tests/result/` are the recorded outputs used to generate the figures and tables in the final report. If regenerated data differs slightly, compare the aggregate trends rather than expecting every trial row to be identical.

## Path Note

The final scripts use paths relative to their own file locations, such as `tests/result/`, rather than machine-specific absolute paths. This means the project folder can be moved to another computer and the scripts should still find their inputs and outputs as long as the folder structure is preserved.
