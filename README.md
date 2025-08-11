# 📈 Linear Regression Techniques in R

[![Made with R](https://img.shields.io/badge/Made%20with-R-1f425f.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project focuses on implementing and comparing various linear regression techniques using R. Developed as part of a graduate-level assignment for the Data Science & Data Analysis course at the University of Salerno, the objective is to identify the best predictive model for a given dataset by minimizing test set error.

> This demonstrates proficiency in R programming, statistical analysis, and reproducible research workflows.

📂 This project is part of a series of regression analysis case studies. For another similar project, see [regression-model-comparison-R](https://github.com/francescopiocirillo/regression-model-comparison-R).

## 📌 Objective

Given a dataset with 60 observations and 30 predictors (X1–X30), the goal is to:

1. Assess correlation and multicollinearity among predictors.
2. Implement OLS manually, without using `lm()` or `summary()`.
3. Compare manual results with built-in R functions.
4. Apply model selection strategies (stepwise, Ridge, Lasso) to minimize test MSE.
5. Identify statistically significant predictors.

## 🌍 Language Note

All code comments and the report are written in Italian, as this project was originally developed in an academic setting in Italy. Nonetheless, the structure, organization, and methodology follow international best practices in data science and statistical modeling.

## 🛠️ Techniques Implemented

- 📉 **Correlation analysis** and **multicollinearity check** using `VIF`
- 🧮 **Manual OLS implementation** (matrix algebra, p-values without `summary()`)
- ⚖️ **Model selection techniques**:
  - Forward selection (Cp, BIC, CV)
  - Backward elimination (Cp, BIC, CV)
  - Hybrid selection
  - Ridge regression
  - Lasso regression
- 🧪 **5-fold Cross-validation**
- 📊 **MSE computation on test set** to compare all models

## 📈 Methodology

- Data split: 80% training / 20% testing
- Models fitted on training set, evaluated on test set
- All MSE values collected and best model selected
- Final model’s predictors and coefficients reported

## 📂 Repository Structure

```
📦 linear-regression-from-scratch-R/
├── 📁 analysis/
│   └── regression_analysis.R              # Complete R script
│
├── 📁 data/
│   └── Regression2024.csv                 # Dataset used for analysis
│
├── 📁 graphs/
│   ├── bwd_adjr2.png                      # Adjusted R² for backward stepwise
│   ├── bwd_bic.png                        # BIC plot for backward stepwise
│   ├── bwd_cp.png                         # Cp plot for backward stepwise
│   ├── bwd_cv_error.png                   # CV error for backward stepwise
│   ├── corrplot_matrice.png              # Correlation matrix visualization
│   ├── cv_forward.png                     # Cross-validation error (forward stepwise)
│   └── hyb_cv_error.png                   # CV error for hybrid selection
│
├── 📁 instructions/
│   ├── project_brief_ENGLISH.pdf          # Assignment prompt (English)
│   └── project_brief_ITALIAN.pdf          # Assignment prompt (Italian)
│
├── 📁 report/
│   └── raw_output_analysis.txt            # Raw analysis output (log, console results)
│
├── LICENSE                                # License file
└── README.md                              # Project overview (this file)
```

## 📊 Technologies

- **Language**: R
- **Libraries**: `glmnet`, `car`, `leaps`, `boot`, `MASS`, `corrplot`, `reshape2`

## 🎓 About the Project

This project was created as part of the *Data Science & Data Analysis* course (2024) for the Master’s Degree in Computer Engineering at the **University of Salerno**.

## 👥 Team – University of Salerno
    
* [@francescopiocirillo](https://github.com/francescopiocirillo)
    
* [@alefaso-lucky](https://github.com/alefaso-lucky)

## 🔍 Key Highlights

- Demonstrates practical knowledge of regression theory and implementation
- Hands-on experience with statistical computing in R
- Capable of end-to-end modeling: from raw matrix algebra to model selection
- Emphasizes explainability and statistical rigor

## 📬 Contacts

✉️ Got feedback or want to contribute? Feel free to open an Issue or submit a Pull Request!

## 📈 SEO Tags

```
regression, model-selection, linear-regression, ridge-regression, lasso, best-subset-selection, stepwise-selection, cross-validation, R, data-science, statistics, predictive-modeling, machine-learning, feature-selection, multicollinearity, MSE, OLS, glmnet, BIC, AIC
```

## 📄 License

This project is licensed under the **MIT License**, a permissive open-source license that allows anyone to use, modify, and distribute the software freely, as long as credit is given and the original license is included.

> In plain terms: **use it, build on it, just don’t blame us if something breaks**.

> ⭐ Like what you see? Consider giving the project a star!
