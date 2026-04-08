# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Environment
Run this to make the environment run and use the ml-course components we need.
```bash 
conda activate ml-course
```  

## Repository Overview

Academic Machine Learning course work for Group 8 (Abelardo, Aquino, Balingit, Gumban), 3CSD section. Each folder corresponds to a module or assessment.

## Folder Structure

| Folder | Content |
|--------|---------|
| `[1] Group 8 - Module 1 - Activity Notebook 1` | Intro activities: moving average, least squares regression, dummy variables, other model types |
| `[2] Group 8 - Module 2 - Predicting Bike Rental Demand with Linear Regression` | Group project on bike rental demand prediction |
| `[3] Group 8 - Module 2 - Linear Regression Lab Work (Summative)` | Summative linear regression lab |
| `[4] Group 8 - Model Evaluation - Assessmen` | Summative on model evaluation using MAGIC Telescope dataset (SVM, AUC-ROC, threshold calibration) |
| `[4] Group 8 - Module 4 - CART Lab Activity` | CART lab covering decision trees, pruning, nested cross-validation |

## Common Libraries Used

- **NumPy, Pandas** — data manipulation
- **Matplotlib** — visualization
- **Scikit-learn** — models (DecisionTree, SVM), metrics, GridSearchCV/RandomizedSearchCV, cross-validation, preprocessing
- **OpenML** — dataset loading (used in model evaluation notebook)

## Working with Notebooks

Open and run notebooks with Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```

Run a specific notebook non-interactively:
```bash
jupyter nbconvert --to notebook --execute "<notebook_path>.ipynb"
```

## Key ML Concepts by Module

- **Module 1:** Time series (moving average), least squares regression, dummy variables
- **Module 2:** Linear regression, feature engineering, bike demand prediction
- **Model Evaluation:** Binary classification metrics, cost-sensitive learning, SVM with RBF kernel, threshold calibration
- **Module 4 (CART):** Decision trees from scratch, Gini/entropy criteria, cost-complexity pruning (1-SE rule), nested cross-validation, hyperparameter tuning
