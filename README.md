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

PW2 - Lab B: Optimization in Chemistry

What I built:
Implemented and compared three numerical optimization algorithms (Gradient Descent, Newton's method, and SLSQP via `scipy.optimize.minimize`) across convex and complex cost landscapes. Applied these optimization techniques to real chemical problems: fitting reaction rate kinetics from experimental data, calculating chemical equilibrium concentrations, and identifying titration equivalence points.

Results & Observations:
- **Optimization Methods Comparison:** On simple convex functions ($f(x) = (x-3)^2 + 1$), all three methods converge easily to $x \approx 3$. On complex landscapes ($g(x) = x^4 - 3x^2 + x + 5$), initial guesses ($x_0$) drastically influence convergence; Newton's method requires checking second-derivative signs ($g'' > 0$) to distinguish minima from local maxima or stationary points.
- **Reaction Rate Fitting:** Fitted first-order kinetics $C(t) = C_0 e^{-kt}$ by minimizing total squared error, obtaining a rate constant $k \approx 0.25 \text{ s}^{-1}$.
- **Chemical Equilibrium:** Solved the equilibrium imbalance equation $\frac{(2x)^2}{(1-x)^2} - K = 0$ using both `scipy.optimize.newton` and SLSQP minimization, yielding identical extent of reaction $x \approx 0.66$ (corresponding to $n_{\text{HI}} \approx 1.33 \text{ mol}$).
- **Titration Equivalence Point:** Located the steepest point of the pH curve using numerical differentiation (`np.gradient`), finding the peak derivative at $V \approx 50 \text{ mL}$.