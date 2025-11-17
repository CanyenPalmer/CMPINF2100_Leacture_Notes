# CMPINF 2100 – Data-Centric Computing Notes & Assignments

This repository collects my lecture notes, weekly practice notebooks, and graded assignments from **CMPINF 2100 (Data-Centric Computing)**.  
It serves both as a **learning journal** (organized by week) and a **portfolio** of hands-on work in data wrangling, exploratory data analysis, and modeling.

---

## 🚀 Tech Stack Overview

<p align="center">
  <!-- Languages & Environment -->
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white" />
  <!-- Data & Numerics -->
  <img src="https://img.shields.io/badge/pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white" />
  <!-- Visualization -->
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0" />
  <!-- Modeling -->
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Toolkit-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/statsmodels-Regression%20%26%20Inference-008080" />
  <img src="https://img.shields.io/badge/SciPy-Scientific%20Computing-8CAAE6?logo=scipy&logoColor=white" />
  <!-- Tools -->
  <img src="https://img.shields.io/badge/Git-GitHub-181717?logo=github&logoColor=white" />
</p>

---

## 🧠 Skills Developed (Employer-Facing Summary)

Over the course of CMPINF 2100, I built a practical foundation in **data science workflows**:

- **Programming & Reproducibility**
  - Writing clear, documented Python code in Jupyter notebooks.
  - Combining markdown and code to explain analysis steps.
  - Organizing work in a version-controlled GitHub repository.

- **Data Wrangling & Feature Preparation**
  - Reading data from CSV and Excel into `pandas` (`pd.read_csv`, `pd.read_excel`).
  - Cleaning and transforming data (type conversion, handling duplicates, filtering, string operations).
  - Summarizing data with groupby, aggregations, and descriptive statistics.

- **Exploratory Data Analysis (EDA)**
  - Visualizing distributions and relationships with `matplotlib` and `seaborn`.
  - Using scatterplots, histograms, boxplots, point plots, trend plots, correlation plots, and pair plots.
  - Interpreting patterns, correlations, and clusters to inform modeling decisions.

- **Statistical & Machine Learning Techniques**
  - Simulation-based reasoning with randomness and averages.
  - K-Means clustering for unsupervised learning.
  - Principal Component Analysis (PCA) for dimensionality reduction.
  - Linear regression (simple and nonlinear forms) using `statsmodels`.
  - Binary classification via logistic regression and classification thresholds.

- **Model Evaluation & Interpretation**
  - Visual diagnostics for linear model assumptions (linearity, residual behavior).
  - Comparing regression models and metrics.
  - Working with accuracy, confusion matrices, and ROC curves for classifiers.
  - Interpreting coefficients, features, and interactions for model storytelling.

- **End-to-End Analysis & Communication**
  - Executing full EDA workflows (problem framing → wrangling → visualization → modeling plan).
  - Combining multiple datasets through joins, merges, and relational reasoning.
  - Communicating findings clearly in markdown cells with supporting visuals.

---

## 📚 Table of Contents

- [Week 01 – Getting Started with Jupyter & Python Basics](#week-01--getting-started-with-jupyter--python-basics)
- [Week 02 – Iteration, Randomness & Simulation](#week-02--iteration-randomness--simulation)
- [Week 03 – Side Effects & Counting Unique Values](#week-03--side-effects--counting-unique-values)
- [Week 05 – Working with Tabular Data in pandas](#week-05--working-with-tabular-data-in-pandas)
- [Week 06 – Introduction to Matplotlib](#week-06--introduction-to-matplotlib)
- [Week 07 – Statistical Visualization with Seaborn](#week-07--statistical-visualization-with-seaborn)
- [Week 08 – Exploratory Data Analysis Project (EDA)](#week-08--exploratory-data-analysis-project-eda)
- [Week 09 – Midterm Exam: Joining Data, Clustering & PCA](#week-09--midterm-exam-joining-data-clustering--pca)
- [Week 10 – Linear Regression & Model Comparison](#week-10--linear-regression--model-comparison)
- [Week 11 – Feature Engineering & Logistic Regression Foundations](#week-11--feature-engineering--logistic-regression-foundations)
- [Week 12 – Classification Metrics & ROC Analysis](#week-12--classification-metrics--roc-analysis)
- [Final Summary & Reflections](#final-summary--reflections)

---

## Week 01 – Getting Started with Jupyter & Python Basics

**Overview**

Week 01 focused on setting up the workflow for the course: working in Jupyter notebooks, understanding the difference between code and markdown cells, and writing simple Python code in script and notebook form. This week laid the foundation for using notebooks as both a computational tool and a communication medium.

**Tech & Tools**

- **Languages & environment:** Python, Jupyter Notebooks
- **Concepts:** markdown vs code cells, printing, basic expressions
- **Imports:** none beyond the Python standard library

**Files in this week**

- `our_first_notebook.ipynb` — Introductory notebook demonstrating markdown cells, headers, and simple Python code execution.
- `our_first_notebook.html` — Exported HTML version of the introductory notebook for easy viewing.
- `our_first_script.py` — A basic Python script illustrating how to write and run code outside of notebooks.
- `hw01.ipynb` — Homework 1 notebook practicing the fundamentals of notebooks, code cells, and simple Python operations.
- `hw01.html` — Exported HTML version of Homework 1.

---

## Week 02 – Iteration, Randomness & Simulation

**Overview**

Week 02 introduced core programming patterns for data work: iterating with `for` loops, using the `random` module to simulate processes, and thinking about averages and distributions through simulation. The emphasis was on understanding how iteration and randomness interact to approximate expectations.

**Tech & Tools**

- **Language & environment:** Python, Jupyter
- **Libraries:** `random`, `matplotlib.pyplot` (for simple plots)
- **Concepts:** `for` loops, iteration variables, sequences, random number generation, simulation of averages

**Files in this week**

- `week_02_intro_randomness.ipynb` — Introduces randomness in Python using the `random` module and explains how random draws behave.
- `week_02_review_forloops.ipynb` — Reviews the structure of `for` loops, iteration variables, and loop bodies with concrete examples.
- `week_02_iterate_randomness.ipynb` — Combines iteration and randomness to simulate random processes repeatedly.
- `week_02_calculate_averages.ipynb` — Uses iteration to compute averages over collections and compare them to simulated expectations.
- `week_02_simulating_the_average.ipynb` — Runs repeated simulations and visualizes how sample averages behave, using basic plots.

---

## Week 03 – Side Effects & Counting Unique Values

**Overview**

Week 03 moved into more subtle Python behavior, including **side effects** of functions, and introduced techniques for counting unique values in collections. This week built comfort with sets, counting patterns, and reasoning carefully about how state changes as code executes.

**Tech & Tools**

- **Language & environment:** Python, Jupyter
- **Concepts:** side effects in functions, sets, counting patterns, list comprehensions
- **Imports:** primarily standard Python built-ins (`set`, `len`, etc.)

**Files in this week**

- `week_02_side_effects.ipynb` — Explores how functions can modify data through side effects and why that matters for debugging and reproducibility.
- `week_03_count.ipynb` — Demonstrates counting occurrences and unique values using `set()`, loops, and related techniques.

---

## Week 05 – Working with Tabular Data in pandas

**Overview**

Week 05 was the first deep dive into **tabular data** using `pandas`. The focus was on reading data from files, inspecting and summarizing series and data frames, combining datasets via concatenation and joins, and filtering data based on string content. This week established the core data-wrangling toolkit.

**Tech & Tools**

- **Libraries:** `pandas`, `numpy`, `seaborn`, `os`
- **Concepts:** reading CSV/Excel files, understanding Series vs DataFrames, summarizing columns and entire tables, concatenation, joins/merges, string-based filtering
- **File formats:** `.csv`, `.xlsx`

**Files in this week**

- `Example_A.csv` — Example dataset used to practice reading CSV files into pandas.
- `Example_B.csv` — Second example CSV for combining and comparing datasets.
- `Example_C.csv` — Additional CSV for multi-table operations and concatenation.
- `Excel_Example_Data.xlsx` — Excel file used to demonstrate reading Excel data with pandas.
- `joined_data.csv` — Example of a joined dataset, used to illustrate merge outputs.
- `week_05_read_data.ipynb` — Introduces reading CSV and Excel files into pandas and inspecting data with `.head()`, `.info()`, and `.describe()`.
- `week_05_summarize_series.ipynb` — Focuses on summarizing individual `Series` objects with descriptive statistics and basic methods.
- `week_05_summarize_dataframe.ipynb` — Summarizes entire DataFrames, exploring column-level statistics and structure.
- `week_05_combine_concat.ipynb` — Demonstrates vertical and horizontal concatenation of DataFrames using `pd.concat`.
- `week_05_combine_joins_or_merge.ipynb` — Shows how to join and merge tables using keys and different join types (inner, left, etc.).
- `filter_pandas_with_strings.ipynb` — Uses string methods and boolean filtering to subset rows based on text content.

---

## Week 06 – Introduction to Matplotlib

**Overview**

Week 06 introduced **matplotlib**, focusing on the difference between the classic stateful (MATLAB-style) interface and the object-oriented approach. The goal was to build basic intuition for figures, axes, and how plots are constructed and customized.

**Tech & Tools**

- **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`
- **Concepts:** stateful vs object-oriented plotting, figures and axes, basic plot creation and configuration

**Files in this week**

- `week_06_matplotlib_intro.ipynb` — Explains the fundamentals of matplotlib, creates simple plots, and distinguishes between figure-level and axes-level operations.

---

## Week 07 – Statistical Visualization with Seaborn

**Overview**

Week 07 expanded visualization skills using **seaborn** for higher-level, statistically-informed plots. The focus was on relationships between variables: continuous-to-continuous via scatterplots and trend plots, and relationships involving categorical variables via boxplots and point plots. Correlation plots and pair plots were introduced for multi-variable exploration.

**Tech & Tools**

- **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`
- **Concepts:** scatterplots for continuous–continuous relationships, trend/line plots, boxplots, point plots, correlation heatmaps, pair plots

**Files in this week**

- `Scatter_Plots.ipynb` — Uses scatterplots to visualize relationships between two continuous variables (e.g., penguin measurements).
- `Trend_Plots.ipynb` — Shows trend lines and smoothed relationships to highlight patterns across continuous variables.
- `Corrplots.ipynb` — Builds correlation matrices and correlation heatmaps to quantify and visualize relationships between multiple variables.
- `PairsPlots.ipynb` — Creates pair plots (scatterplot matrices) to explore all pairwise relationships among several features.
- `Review Boxplots.ipynb` — Revisits boxplots as a way to compare distributions across categories.
- `Review Point Plots.ipynb` — Uses point plots to summarize means and confidence intervals across categorical groups.

---

## Week 08 – Exploratory Data Analysis Project (EDA)

**Overview**

Week 08 centered on a **full exploratory data analysis (EDA) project**, pulling together many of the skills developed in earlier weeks. The assignment guided problem framing (regression vs classification), identification of inputs and outputs, careful data cleaning, and exploratory visualizations to motivate a modeling plan.

**Tech & Tools**

- **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn.preprocessing.StandardScaler`, `sklearn.cluster.KMeans`
- **Concepts:** project-style EDA workflow, identifying targets and inputs, handling missing values and types, deriving response variables, detecting duplicates, clustering and scaling as part of exploration

**Files in this week**

- `Palmer_Canyen_EDA.ipynb` — Full EDA project notebook for the course, combining narrative, code, and visualizations.

### Capstone: EDA Project Details

In `Palmer_Canyen_EDA.ipynb`, I:

- **Framed the predictive task** by deciding whether the problem is regression or classification and clearly specifying inputs vs outputs.
- **Identified and derived targets**, including any necessary summarization (e.g., aggregating rows to define outcome variables).
- **Separated identifiers** (columns not suitable as model inputs) from predictive features.
- **Performed systematic cleaning**, including handling duplicates, missing values, and type conversions using references from the pandas documentation.
- **Explored relationships** using histograms, scatterplots, boxplots, and correlation analysis to spot key drivers and potential interactions.
- **Applied scaling and clustering** (`StandardScaler`, `KMeans`) as part of understanding structure in the data.
- **Documented modeling implications**, summarizing how the EDA informs which features and transformations would make sense in downstream predictive models.

This notebook functions as an end-to-end example of turning a raw dataset into a model-ready understanding.

---

## Week 09 – Midterm Exam: Joining Data, Clustering & PCA

**Overview**

Week 09 featured the **midterm exam**, which served as a comprehensive assessment of data wrangling and analysis skills. The exam required joining multiple CSV files, summarizing and visualizing data, and applying clustering and PCA to understand structure and risk patterns in the data. Additional lecture notebooks reinforced linear model assumptions and unsupervised learning concepts.

**Tech & Tools**

- **Libraries:** `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn.preprocessing.StandardScaler`, `sklearn.cluster.KMeans`, `sklearn.decomposition.PCA`, `scipy.cluster.hierarchy`
- **Concepts:** joining multiple tables, grouping and summarizing, clustering, PCA, hierarchical clustering, linear model assumptions

**Files in this week**

- `Palmer_Canyen_Midterm.ipynb` — Midterm exam notebook integrating joins, summaries, visualizations, clustering, and PCA on machine test data.
- `midterm_machine_01.csv` — Machine-level data source (part 1) used in the midterm.
- `midterm_machine_02.csv` — Machine-level data source (part 2) used in the midterm.
- `midterm_machine_03.csv` — Machine-level data source (part 3) used in the midterm.
- `midterm_supplier.csv` — Supplier or batch-level data used for joining and comparisons.
- `midterm_test.csv` — Test dataset used for evaluating models or summaries built from the training machines.
- `week_09_lm_assumptions.ipynb` — Reviews key assumptions of linear models using visual diagnostics.
- `week_09_lm_linearity.ipynb` — Focuses on checking linearity assumptions between predictors and outcomes.
- `week_09_review_cluster.ipynb` — Recaps K-Means clustering, scaling, and visualization of cluster structure.
- `week_09_review_pca.ipynb` — Revisits PCA to reduce dimensionality and interpret principal components.

### Capstone: Midterm Exam Details

In `Palmer_Canyen_Midterm.ipynb`, I:

- **Joined multiple data sources** from machine and supplier CSVs to build a unified analysis dataset.
- **Computed summaries and group-level metrics** (e.g., failure rates, defect counts) to characterize machine and supplier performance.
- **Visualized key relationships** using matplotlib and seaborn to explore how different features relate to failure risk.
- **Standardized features** and applied **K-Means clustering** to identify groups of machines or cases with similar profiles.
- Used **PCA** to reduce dimensionality, interpret combinations of features, and visualize clusters in lower-dimensional space.
- Documented interpretations and conclusions in markdown, connecting analytical results back to the exam’s manufacturing/test context.

This exam notebook demonstrates an end-to-end workflow that strongly resembles real-world data science tasks: combining data, exploring it, segmenting it, and drawing actionable conclusions.

---

## Week 10 – Linear Regression & Model Comparison

**Overview**

Week 10 focused on **linear regression**, both in simple linear form and with nonlinear relationships. The notebooks cover fitting regression models, generating predictions, evaluating model fit, and comparing multiple models. The week also included review of linear model theory and practice.

**Tech & Tools**

- **Libraries:** `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `statsmodels.formula.api`
- **Concepts:** linear vs nonlinear relationships, training regression models, generating predictions, regression metrics, comparing multiple models

**Files in this week**

- `week_10_linear_data.csv` — Dataset used to illustrate linear relationships and regression fitting.
- `week_10_nonlinear_data.csv` — Dataset with nonlinear structure to motivate flexible modeling.
- `week_10_lm_fitting_linear.ipynb` — Fits linear regression models to linearly-related data using `statsmodels`.
- `week_10_lm_fitting_nonlinear.ipynb` — Explores fitting models to nonlinear patterns and evaluating their performance.
- `week_10_lm_predictions_linear.ipynb` — Generates and visualizes predictions from linear models, including fitted lines.
- `week_10_lm_predictions_nonlinear.ipynb` — Visualizes predictions from models on nonlinear data, highlighting fit vs structure.
- `week_10_regression_metrics.ipynb` — Introduces and computes regression performance metrics to assess model accuracy.
- `week_10_regression_multiple_models.ipynb` — Compares multiple regression models on the same data to understand trade-offs.
- `week_10_review_lm_linear.ipynb` — Reviews core ideas and implementation details of linear regression.
- `week_10_review_lm_nonlinear.ipynb` — Reviews strategies for modeling nonlinear relationships within a linear modeling framework.

---

## Week 11 – Feature Engineering & Logistic Regression Foundations

**Overview**

Week 11 transitioned from regression to **classification**, emphasizing **feature engineering** and the foundations of logistic regression. The focus was on additive vs interaction features, using categorical inputs, and preparing data for models with multiple inputs. Logistic regression was introduced through `statsmodels`, including how to interpret coefficients and link probabilities to linear predictors.

**Tech & Tools**

- **Libraries:** `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `statsmodels.formula.api`
- **Concepts:** additive features, categorical feature encoding, interaction terms, multiple-input models, binary classification framing, logistic regression fitting and interpretation

**Files in this week**

- `linear_additive_example.csv` — Dataset used to illustrate additive linear features in regression-style models.
- `week_11_categorical_input.csv` — Dataset containing categorical inputs for demonstrating encoding and modeling.
- `week_11_features_additive.ipynb` — Explores constructing and interpreting additive feature structures.
- `week_11_features_categorical.ipynb` — Demonstrates working with categorical inputs and how they enter models.
- `week_11_features_interaction.ipynb` — Introduces interaction terms and shows how combinations of features can be modeled.
- `week_11_features_more_inputs_intro.ipynb` — Shows how to extend from single-input to multi-input models.
- `week_11_intro_binary_classification.csv` — Binary classification dataset used for logistic regression examples.
- `week_11_logistic_fitting_statsmodels.ipynb` — Fits logistic regression models using `statsmodels`, interpreting coefficients and probabilities.
- `week_11_logistic_notes.ipynb` — Conceptual notes and examples that explain logistic regression theory and usage.

---

## Week 12 – Classification Metrics, ROC Analysis & Model Comparison

**Overview**

Week 12 focused on **evaluating and comparing classification models**, especially logistic regression. Building on the binary classification setup from Week 11, this week introduced practical tools for judging model quality: accuracy, confusion matrices, ROC curves, and comparisons across multiple models. The emphasis was on understanding the trade-offs created by different probability thresholds and how to choose models and cutoffs that align with real-world objectives.

**Tech & Tools**

- **Libraries:** `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `statsmodels.formula.api`
- **Concepts:**
  - Binary classification workflow: from predicted probabilities to hard labels
  - Accuracy and error decomposition with confusion matrices
  - ROC curves and the trade-off between true positives and false positives
  - Comparing multiple classification models on the same task
  - Impact of different thresholds on performance metrics

**Files in this week**

- `week_12_binary_classification.csv` — Dataset used as the common ground for evaluating and comparing binary classification models.
- `week_12_logistic_classification.ipynb` — Trains logistic regression models, generates predicted probabilities, and sets up the classification task for evaluation.
- `week_12_classification_accuracy.ipynb` — Computes and analyzes classification accuracy under different probability thresholds, highlighting how model performance changes as the cutoff moves.
- `week_12_classification_confusion.ipynb` — Builds confusion matrices to break down errors into true/false positives and true/false negatives, providing a more nuanced view than accuracy alone.
- `week_12_classification_roc.ipynb` — Plots ROC curves and studies how sensitivity and specificity trade off across thresholds; introduces the idea of ranking models by their ROC behavior.
- `week_12_classification_multiple_models.ipynb` — Compares multiple classification models (e.g., different feature sets or model specifications) on the same binary task using metrics and ROC-style evaluation to understand which modeling choices generalize best.

---

## Week 13 – Cross-Validation & Logistic Regression with scikit-learn

**Overview**

Week 13 focused on making classification models **more reliable and trustworthy** by introducing **cross-validation** and deepening practice with **logistic regression** in both `scikit-learn` and `statsmodels`. Building on the binary classification and evaluation tools from Weeks 11–12, this week emphasized how to estimate out-of-sample performance, avoid overfitting, and compare models more systematically using k-fold cross-validation.

**Tech & Tools**

- **Libraries**
  - `pandas`, `numpy`
  - `matplotlib.pyplot`, `seaborn`
  - `scikit-learn`:
    - `sklearn.linear_model.LogisticRegression`
    - `sklearn.model_selection.train_test_split`
    - `sklearn.model_selection.cross_val_score`
  - `statsmodels` / `statsmodels.formula.api` for logistic regression with richer summaries

- **Concepts**
  - Train/test splits vs k-fold cross-validation
  - Estimating generalization performance and reducing variance in evaluation
  - Logistic regression in `scikit-learn` vs `statsmodels`
  - Working with predicted probabilities, decision functions, and class labels
  - Comparing models and feature sets using cross-validated scores

**Files in this week**

- `week_13_cv_intro.ipynb` — Conceptual and practical introduction to cross-validation, showing how different splits of the data can lead to different performance estimates and why averaging across folds provides a more stable view of model quality.
- `week_13_sklearn_logistic.ipynb` — Uses `scikit-learn`’s `LogisticRegression` to fit binary classification models, generate predictions and predicted probabilities, and connect these outputs back to the evaluation metrics introduced in earlier weeks.
- `week_13_cv_logistic_sklearn.ipynb` — Applies k-fold cross-validation to logistic regression in `scikit-learn`, demonstrating how to compute cross-validated accuracy (or other metrics) and interpret the distribution of scores across folds.
- `week_13_cv_logistic_statsmodels.ipynb` — Bridges `statsmodels` and cross-validation by fitting logistic models in `statsmodels` while still evaluating them with a cross-validation-style workflow, highlighting the trade-off between rich statistical summaries and predictive validation.
- `week_13_cv_sklearn_cross_val_score.ipynb` — Uses `sklearn.model_selection.cross_val_score` to streamline the cross-validation process, showing how to quickly compare model specifications or feature sets based on repeated, out-of-sample performance estimates.

---

## Final Summary & Reflections

Across these weeks, this course guided me from **basic Python and notebook usage** to **full EDA projects and model evaluation**. The notebooks in this repository demonstrate:

- A consistent, notebook-based workflow for exploratory analysis.
- Practical experience with `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, and `statsmodels`.
- End-to-end data work: from raw CSVs through cleaning, visualization, modeling, and interpretation.
- A progression from simple loops and randomness to clustering, PCA, regression, and classification metrics.

This repository is both a personal reference and a portfolio snapshot of my applied data science training in CMPINF 2100.
