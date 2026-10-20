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

PW3 - Data, Distributions, Testing, and Causation[cite: 1]

What I built:
Loaded and processed medical and chemical exposure datasets using `pandas` and `scipy.stats`[cite: 1, 2]. Visualized continuous variable distributions, evaluated their normality via statistical testing (Shapiro-Wilk), analyzed mean differences with uncertainty, evaluated correlations, and identified a true causal factor by isolating a confounding variable[cite: 1, 2].

Session 1: Data Characterization[cite: 1]
- **Data Overview:** Loaded `heart.csv` containing ~300 patient observations across 14 clinical attributes[cite: 1]. Focused on four continuous variables: `age`, `chol`, `trestbps`, and `thalach`[cite: 1].
- **Distribution Shapes:** Visualized distributions using histograms (`distributions.png`)[cite: 1]. `age` appears roughly symmetric, `chol` and `trestbps` are right-skewed with long right tails, and `thalach` exhibits a left-skewed shape[cite: 1].
- **Normality Check:** Performed Shapiro-Wilk normality tests (`scipy.stats.shapiro`)[cite: 1]. All four variables yielded $p < 0.05$, demonstrating significant departures from a normal distribution[cite: 1]. Non-parametric statistical methods were selected for subsequent hypothesis testing based on these verdicts[cite: 1, 2].

Session 2: Statistical Inference & Causation[cite: 1, 2]
- **Group Heart Rate Comparison:** Split `thalach` (max heart rate) by `target` (disease vs. healthy)[cite: 1, 2]. Applied a Mann-Whitney U test due to non-normality, obtaining $p < 0.001$[cite: 1, 2]. Healthy patients had a lower mean (`mean` $\approx 139$ bpm) compared to diseased patients (`mean` $\approx 158$ bpm)[cite: 2]. Plotted group means with standard error errorbars (`thalach_comparison.png`), confirming a statistically significant separation[cite: 2].
- **Age vs. Heart Rate:** Computed Pearson/Spearman correlation between `age` and `thalach`, obtaining $r \approx -0.39$ ($p < 0.001$)[cite: 2]. Plotted `age_vs_thalach.png`, showing a moderate negative relationship where maximum heart rate decreases with age[cite: 2].
- **Chemical Exposure Mystery:** Naive correlation on `chemicals_cancer.csv` suggested `benzene` was strongly associated with tumour `malignancy` ($r \approx 0.70$) while `cadmium` appeared weakly correlated ($r \approx 0.10$)[cite: 2].
- **Confounding Variable Analysis:** Controlled for `pollution_index` by filtering rows where $40 < \text{pollution\_index} < 60$[cite: 2]. Upon controlling for pollution, the association between `benzene` and `malignancy` vanished ($r \approx 0$), whereas `cadmium` maintained its positive relationship[cite: 2]. `benzene` was a spurious correlation caused by the confounding pollution index, identifying `cadmium` as the true driver of malignancy[cite: 2].

Bonus: Category Unpredictability[cite: 2]
- **Target Entropy:** Evaluated class balance for `target` using normalized value counts[cite: 2]. Calculated Shannon entropy $H = -\sum p_i \log_2(p_i) \approx 0.99 \text{ bits}$ (close to the maximum $1.0 \text{ bit}$), indicating a near-balanced, highly unpredictable distribution[cite: 2].

Conclusion:
Observational correlations can be misleading due to confounding variables, requiring controlled subset filtering or multivariable adjustment to distinguish real causal relationships from spurious associations[cite: 2].