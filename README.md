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

Lab B: Data, Plotting, and Automation

What I built:
Loaded real decay observation data from CSV, plotted a side-by-side comparison between experimental data and the analytical decay law, and automated the visualization pipeline using Snakemake.

Results & Observations:
    Data match: The observed scatter points perfectly follow the theoretical exponential curve ($N_0 e^{-\lambda t}$ with $\lambda = 0.3$).

    Pipeline: Snakemake tracks file timestamps and automatically rebuilds figure.png only when inputs change.

Conclusion:
Comparing real data with analytical models side-by-side validates physical expectations. Automating figure generation with Snakemake ensures workflow reproducibility and avoids unnecessary re-computations.