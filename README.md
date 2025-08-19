# Linear Regression from Scratch in R — OLS, Ridge & Lasso 📊

[![Releases](https://img.shields.io/github/v/release/ruhsigadilkar/linear-regression-from-scratch-R?color=blue&label=Download%20Release)](https://github.com/ruhsigadilkar/linear-regression-from-scratch-R/releases)

![R logo](https://www.r-project.org/logo/Rlogo.png)

Table of Contents
- About
- Key features
- Dataset
- What you get
- How to run (quick)
- Manual OLS: method and p-values
- Model selection: Stepwise (AIC/BIC)
- Regularization: Ridge and Lasso
- Cross-validation and evaluation
- Results and interpretation
- File structure
- Examples
- How to extend
- Contributing
- License
- Releases

About
This project implements linear regression end-to-end in R using a dataset with 30 predictors. It shows how to build an ordinary least squares (OLS) estimator without using lm(), compute p-values from theory, and compare results with built-in functions. The repo applies stepwise selection with AIC and BIC, fits Ridge and Lasso via glmnet, and uses cross-validation to minimize test error and identify key predictors.

Key features
- Manual OLS implemented using matrix algebra (no lm()).
- Closed-form parameter estimates and analytical standard errors.
- p-value computation from t-distribution for each coefficient.
- Validation against lm() output for correctness.
- Backward and forward stepwise selection using AIC and BIC.
- Ridge and Lasso regression with glmnet and cross-validation.
- k-fold cross-validation loops for robust test error estimates.
- Diagnostics for multicollinearity, influence, and residuals.
- Scripts and functions that reproduce figures and tables.

Dataset
The demo dataset contains 30 numeric predictors and one continuous response. The data simulates correlated predictors to demonstrate multicollinearity. The repo includes a reproducible data generator and a sample CSV for quick testing.

What you get
- R scripts that run the full pipeline.
- A data generator to reproduce experiments.
- Plots: residuals, coefficient paths, CV error curves, and VIFs.
- Comparison tables for OLS vs. lm() vs. regularized fits.
- A README with step-by-step commands and design notes.

How to run (quick)
1. Download the release asset and execute the included R script to reproduce the analysis and figures. The release bundle contains the required R scripts and sample data; download it and run the main script in R or from the shell. For example:
   - Download the release file from Releases and extract it.
   - From shell: `Rscript run_analysis.R`
   - From R: `source("run_analysis.R")`
2. The release file must be downloaded and executed to reproduce the results. See Releases: https://github.com/ruhsigadilkar/linear-regression-from-scratch-R/releases

Manual OLS: method and p-values
This section explains the math implemented in the code and the steps the scripts follow.

Model
- y = Xβ + ε, with ε ~ N(0, σ²I)
- X has an intercept column and 30 predictors.

Estimation
- β_hat = (X'X)^{-1} X'y
- Residuals: e = y − Xβ_hat
- σ_hat² = (e'e) / (n − p)

Standard errors and t-statistics
- Var(β_hat) = σ_hat² (X'X)^{-1}
- se(β_j) = sqrt(diag(Var(β_hat)))
- t_j = β_hat_j / se(β_j)
- p_j = 2 * pt(−|t_j|, df = n − p)

The code computes these values directly with matrix operations. The repo includes a comparison script that calls lm() and compares coefficients, standard errors, t-stats, and p-values. If the design matrix is singular or near-singular, the scripts handle it by reporting condition numbers and using a pseudo-inverse for stability.

Model selection: Stepwise (AIC/BIC)
The repository implements both forward and backward stepwise selection.

Algorithm
- Start from intercept-only or full model.
- At each step, evaluate candidate additions or removals.
- Use AIC and BIC as selection criteria.
- Stop when AIC/BIC no longer improves.

Implementation details
- We compute AIC = 2k − 2 log L and BIC = log(n) k − 2 log L, using the usual Gaussian log-likelihood.
- The scripts report the selected model for AIC and for BIC separately.
- The code tracks model size vs. validation error to show the bias-variance trade-off.

Regularization: Ridge and Lasso
The repo uses glmnet for Ridge and Lasso, and it shows how to prepare data for glmnet.

Workflow
- Standardize predictors before regularized fitting.
- Use cv.glmnet for k-fold cross-validation across lambda.
- For Ridge, use alpha = 0; for Lasso, use alpha = 1.
- Extract lambda.min and lambda.1se and evaluate test error.

Coefficient paths and interpretation
- The code plots coefficient paths as lambda varies.
- It identifies predictors that become zero under Lasso.
- It compares selected predictors vs. stepwise and OLS.

Cross-validation and evaluation
Cross-validation is central to the project. The repo implements:
- k-fold cross-validation (default k = 10).
- Repeated CV for stability (default repeats = 5).
- Train/validation/test splits for final evaluation.

Metrics
- Mean squared error (MSE)
- Root MSE (RMSE)
- R-squared for test data
- Mean absolute error (MAE)
All metrics report mean and standard error across folds or repeats.

Diagnostics and multicollinearity
- Variance Inflation Factor (VIF) calculations to detect multicollinearity.
- Condition number of X'X to detect near-singularity.
- Cook's distance and leverage plots to flag influential points.
- Residual diagnostics: QQ plot, heteroskedasticity tests.

Results and interpretation
The scripts produce a results folder with:
- A table comparing OLS beta_hat vs. lm() and glmnet coefficients.
- P-values for OLS manual implementation.
- AIC- and BIC-selected models and their test RMSE.
- Ridge and Lasso test RMSE and selected features.

Typical findings shown by the code
- Lasso tends to select a sparse set of predictors that reduce test error.
- Ridge shrinks coefficients and reduces variance when predictors correlate.
- Stepwise selection with AIC often keeps more variables than BIC.
- Manual OLS matches lm() numeric output when X'X is well-conditioned.

File structure
- data/
  - sample_data.csv
  - generate_data.R
- scripts/
  - run_analysis.R           # orchestrates the whole pipeline
  - manual_ols.R             # matrix-based OLS and p-values
  - compare_with_lm.R        # side-by-side checks
  - stepwise_selection.R     # forward/backward AIC/BIC
  - ridge_lasso_glmnet.R     # glmnet wrappers and plots
  - cv_utils.R               # cross-validation helpers
  - diagnostics.R            # VIF, Cook's distance, plots
- results/
  - figures/
  - tables/
- README.md
- LICENSE

Examples
Reproduce the main analysis:
1. Download and extract the release bundle from Releases and run:
   - `Rscript scripts/run_analysis.R`
2. Or run parts interactively:
   - Source the manual OLS script: `source("scripts/manual_ols.R")`
   - Fit glmnet: `source("scripts/ridge_lasso_glmnet.R")`

Quick R snippet to run manual OLS interactively
```r
source("scripts/manual_ols.R")
data <- read.csv("data/sample_data.csv")
fit <- manual_ols(data$y, data[, paste0("x", 1:30)])
print(fit$coefficients)
print(fit$p_values)
```

How to extend
- Add categorical predictors with one-hot encoding in the data generator.
- Replace the Gaussian noise generator with a heavy-tailed noise distribution to test robustness.
- Add Elastic Net (glmnet alpha between 0 and 1) and compare with Lasso and Ridge.
- Implement bootstrapped confidence intervals for coefficients.
- Parallelize repeated CV with foreach for speed.

Contributing
- Open an issue to request features or report bugs.
- Fork, implement changes, and create a pull request.
- Keep commits small and add tests for new scripts.

License
- MIT License. See LICENSE file.

Releases
Download the release asset, extract it, and execute the included R script(s) to reproduce the analysis. The release bundle contains scripts, sample data, and a prebuilt results folder for quick inspection. Visit the Releases page to pick a file and follow the instructions inside the bundle:
- https://github.com/ruhsigadilkar/linear-regression-from-scratch-R/releases

Topics and tags
cross-validation, data-science, feature-selection, lasso-regression, linear-regression, machine-learning, model-selection, multicollinearity, predictive-modeling, r, regression, ridge-regression, statistics, stepwise-regression

References and useful links
- R project: https://www.r-project.org/
- glmnet: https://cran.r-project.org/package=glmnet
- Matrix algebra reference: any standard linear algebra text for OLS derivations

Contact
Report issues via the GitHub repository issues page.