# CSPC Computer Science for Physics and Chemistry
My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup
Create the environment for a given lab:
```bash
conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc

PW1

Lab A: Reproducible Foundations

What I built:
Set up the lab repository structure, configured the Conda environment, integrated decay simulation algorithms, added unit tests with pytest, and measured performance speedup of NumPy vectorization.

Speed comparison (loop vs NumPy):

    loop: 3.0855 s

    numpy: 0.0002 s

    speed-up: 12485.47x faster

Tests: all passing? yes

Conclusion:
Using Conda ensures complete environment reproducibility across machines. Vectorizing operations with NumPy dramatically speeds up calculations compared to Python loops. Unit testing with pytest verifies physical validity.

PW1 - Lab B: Data, Plotting, and Automation

Results & Analysis
- **Data Observation:** Dataset `decay_observed.csv` demonstrates exponential decay behavior over time.
- **Comparison:** The experimental scatter plot aligns extremely well with the analytical decay law $N(t) = N_0 e^{-\lambda t}$ using $\lambda = 0.3$, confirming theoretical expectations.
- **Pipeline Automation:** A Snakemake rule was configured to manage figure generation, triggering recalculations automatically only when dependency files are modified.

PW2 Lab A: Motion from Tracking Data

What I built:
Loaded noisy free-fall tracking data from CSV, calculated velocity and acceleration using numerical differentiation (`numpy.gradient`), integrated back to recover position using `scipy.integrate.cumulative_trapezoid`, and generated a three-panel plot comparing position, velocity, and acceleration.

Results & Observations:
Mean Acceleration: -9.81 m/s² (matches theoretical gravity $g \approx 9.81 \text{ m/s}^2$).

Noise Effect: Differentiation amplifies noise, causing wild fluctuations in acceleration even with smooth position data. Integration suppresses noise, recovering the original trajectory within ~1 m.

Conclusion:
Numerical differentiation amplifies measurement noise while integration suppresses it, demonstrating that accumulating data cancels out random errors while calculating rates of change magnifies them.


#test