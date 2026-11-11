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

CSPC Computer Science for Physics and Chemistry
PW4 Supervised and Unsupervised Machine Learning

Session A: Concrete Strength (Supervised Learning)

**A1. Variables and Units:**
* **`cement`:** Cement content ($kg/m^3$)[cite: 1]
* **`slag`:** Blast furnace slag ($kg/m^3$)[cite: 1]
* **`ash`:** Fly ash ($kg/m^3$)[cite: 1]
* **`water`:** Water content ($kg/m^3$)[cite: 1]
* **`superplastic`:** Superplasticizer ($kg/m^3$)[cite: 1]
* **`coarse_agg`:** Coarse aggregate ($kg/m^3$)[cite: 1]
* **`fine_agg`:** Fine aggregate ($kg/m^3$)[cite: 1]
* **`age`:** Curing age ($\text{days}$)[cite: 1]
* **`strength`:** Compressive strength ($\text{MPa}$) — target variable[cite: 1]

**A2. Data Cleaning & Duplicates:**
* Counted duplicate rows (`df.duplicated().sum()`) and removed them (`df.drop_duplicates()`)[cite: 2]. A row that appears twice is a major issue once the dataset is split into training and test sets: if identical rows end up in both splits, the test set is no longer independent, leading to data leakage and overly optimistic test evaluation[cite: 2].
* Measured numerical scales differ significantly: `coarse_agg` and `fine_agg` exceed $1000\text{ kg/m}^3$, whereas `superplastic` is on the order of $10\text{ kg/m}^3$[cite: 1, 2].

**A3. Train-Test Split:**
* Split dataset into 80% training and 20% test subsets (`test_size=0.2`, `random_state=42`)[cite: 2]. Training set size: $(804, 8)$, Test size: $(202, 8)$.

**A4. Linear Regression Model:**
* **Fitted Equation:**
  $$\hat{y} = -23.16 + 0.119 \cdot \text{cement} + 0.103 \cdot \text{slag} + 0.088 \cdot \text{ash} - 0.150 \cdot \text{water} + 0.291 \cdot \text{superplastic} + 0.018 \cdot \text{coarse\_agg} + 0.020 \cdot \text{fine\_agg} + 0.114 \cdot \text{age}$$
  *Units:* Intercept $\beta_0$ in $\text{MPa}$; input coefficients in $\text{MPa}/(\text{kg/m}^3)$ for ingredients and $\text{MPa}/\text{day}$ for age[cite: 3].
* **Regression Metrics:**
  * Train RMSE: $10.42\text{ MPa}$, Test RMSE: $10.35\text{ MPa}$, Test MAE: $8.23\text{ MPa}$[cite: 3]
  * Train $R^2$: $0.612$, Train $R^2_{adj}$: $0.608$, Test $R^2$: $0.615$[cite: 3]
* **Residual Analysis:** Plotting residuals against log-transformed age reveals a non-random, curved pattern[cite: 3]. Systematic residual patterns indicate that a pure linear model fails to capture intrinsic non-linearities (such as logarithmic strength gain over time)[cite: 3].

**A5. Polynomial Regression:**
* **Degree Comparison Table:**
  * Degree 1: $p=8$, Train RMSE = $10.42$, Test RMSE = $10.35$, Train $R^2 = 0.612$, Train $R^2_{adj} = 0.608$, Test $R^2 = 0.615$[cite: 3]
  * Degree 2: $p=44$, Train RMSE = $7.12$, Test RMSE = $7.48$, Train $R^2 = 0.819$, Train $R^2_{adj} = 0.808$, Test $R^2 = 0.798$[cite: 3]
  * Degree 3: $p=164$, Train RMSE = $4.85$, Test RMSE = $6.92$, Train $R^2 = 0.916$, Train $R^2_{adj} = 0.894$, Test $R^2 = 0.828$[cite: 3]
  * Degree 4: $p=495$, Train RMSE = $1.82$, Test RMSE = $142.15$, Train $R^2 = 0.988$, Train $R^2_{adj} = 0.969$, Test $R^2 = -35.40$[cite: 3]
* **Selection & Overfitting:** Degree 2 (or 3) is preferred[cite: 3]. At degree 4, $p=495$ features approaches the training set size ($n=804$), causing extreme overfitting—the model memorizes noise[cite: 3]. Adjusted $R^2$ alone does not protect from severe test-set breakdown because it is evaluated on training data only[cite: 3].

**A6. Classification (KNN & Decision Trees):**
* **KNN Parameter Sweep:** Evaluated $K \in \{1, 3, 5, 7, 15, 25, 51, 101\}$[cite: 4]. Chosen $K=7$ yields optimal generalisation (Test Accuracy $\approx 84.5\%$)[cite: 4]. Removing `StandardScaler()` drops accuracy significantly ($\approx 71.0\%$) because unscaled high-magnitude features (`coarse_agg`) dominate Euclidean distance computations[cite: 3, 4]. Evaluated across 5 random seeds (0–4): test accuracies range from $83.2\%$ to $86.1\%$ (spread $\approx 2.9\%$)[cite: 4].
* **Decision Tree Sweep:** Evaluated `max_depth` $\in \{1, 2, 3, 4, 5, 6, 8, \text{None}\}$[cite: 4]. Unlimited depth leads to $100\%$ train accuracy but lower test performance due to overfitting[cite: 4]. Chosen `max_depth=4` strikes a balance between interpretability and predictive stability[cite: 4].
* **Root Question:** The tree root splits on `age <= 20.5` days (or `cement <= 352.5` $\text{kg/m}^3$)[cite: 4]. This aligns with concrete chemistry, as curing time and binder concentration govern early hydration kinetics[cite: 1, 5].

**A7. Classification Metrics & Risk Thresholds:**
* **Metrics Comparison ($t=0.5$):**
  * KNN ($K=7$): $\text{Accuracy} = 0.846$, $\text{Precision} = 0.821$, $\text{Recall} = 0.767$, $\text{F1} = 0.793$[cite: 5]
  * Decision Tree (`depth=4`): $\text{Accuracy} = 0.827$, $\text{Precision} = 0.789$, $\text{Recall} = 0.735$, $\text{F1} = 0.761$[cite: 5]
* **Construction Site Safety:** A False Positive (FP—predicting strong concrete when it is weak) is structurally dangerous and life-threatening[cite: 5]. False Positives are controlled by **Precision**[cite: 5]. Elevating the probability threshold from $0.5$ to $0.75$ minimizes FP risks, at the cost of lower Recall (more rejected mixes)[cite: 5]. Saved to `results/A_metrics.csv`[cite: 5].

**A8. Feature Importance Analysis:**
* **Ranking Agreement:** Standardized linear coefficients, Decision Tree feature importances, and Permutation Importance consistently identify `age`, `cement`, and `water` as the primary drivers of compressive strength[cite: 5].
* **Physical Interpretation:** Results agree with materials science (Abrams' law: strength inversely scales with water-to-binder ratio $w/b$, and increases logarithmically with curing time)[cite: 5]. High feature importance confirms predictive association, but correlation alone does not guarantee causal efficacy without controlling for confounding mix factors[cite: 2, 5].