# Syllabus companion notebooks

Six executed notebooks, one per syllabus unit, all on the UCI Adult data the TaskClean research uses. They are
**additive**: they read `data/adult_clean.csv` and import a few functions from `src/` read-only (`three_way_split`,
`build_preprocessor`, `build_models`, `evaluate`, `detect_*`). Nothing in `src/`, `results/` or `app.py` is modified, so the
frozen research phases are untouched. Notebook 5 reproduces `results/phase2_clean_baseline.csv` exactly (max difference 0.0000).

## Run them

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-notebook.txt
```

Open a notebook in VS Code and select the `.venv` interpreter as the kernel. Each notebook starts with a small cell that finds
`common.py`, so it works whether the working folder is `notebooks/` or the project root.

## Concept map: syllabus topic -> where it is covered

`TC` = already covered by the TaskClean code itself; `NB` = covered in the notebooks (section number given).

### Unit I - Introduction ([unit1_introduction.ipynb](unit1_introduction.ipynb))
| Syllabus topic | Where |
|---|---|
| Definition of data science, Big Data hype, datafication, landscape of perspectives | NB 1.1 (theory) |
| Statistical inference | NB 1.3 (confidence-interval coverage, Welch t-test, chi-square, effect sizes); TC `phase11_readiness.py` (Pearson/Spearman with p-values) |
| Populations and samples | NB 1.2 (sampling distribution, CLT, biased sample); TC `data.py` stratified split, `taskclean.py` stratified sampling |
| Probability distributions | NB 1.4 (normal fit, Q-Q plot, Shapiro-Wilk, binomial vs simulation) |
| Statistical modelling, fitting a model | NB 1.5 (logistic regression odds ratios); TC `baseline.py`, `training_log.py` |
| Overfitting | NB 1.6 (train vs validation accuracy against tree depth, unlimited vs limited forest) |
| Python environment for data science | NB 1.7 (venv, libraries, versions) |

### Unit II - Data types and statistical description ([unit2_data_types_statistics.ipynb](unit2_data_types_statistics.ipynb))
| Syllabus topic | Where |
|---|---|
| Attributes: nominal, binary, ordinal, numeric, asymmetric, discrete vs continuous, number of values | NB 2.1 (every attribute typed and checked against the data) |
| Mean, median, mode | NB 2.2 (by hand, then pandas; trimmed and weighted mean) |
| Range, quartiles, variance, standard deviation, IQR | NB 2.3 (variance by hand, five-number summary, 68-95-99.7 check); TC `detect.py` IQR fences |
| Graphic displays of basic statistical descriptions | NB 2.4 (histogram, boxplot, ECDF, bar chart) |

### Unit III - Data cleaning and preparation ([unit3_cleaning_preparation.ipynb](unit3_cleaning_preparation.ipynb))
| Syllabus topic | Where |
|---|---|
| Structural inspection | NB 3.1 (`info`, duplicates, placeholders); TC `taskclean.load_csv` (encoding/delimiter/decimal sniffing) |
| Data sources and data quality | NB 3.2 (rule-based quality report); TC `quality_report.py`, six detectors |
| Data cleaning | NB 3.3 (inject known damage, detect with the project's detectors, grade, repair, check); TC `phase9_repair_engine.py`, `taskclean.py` |
| Identifying outliers; measures of centre and spread | NB 3.4 (z-score vs IQR vs percentile, robustness of mean vs median); TC `detect_outliers_iqr` |
| Hypothesis testing vs EDA | NB 3.5 (multiple-comparisons simulation, explore-on-half / confirm-on-half) |
| Data transformation | NB 3.6 (log, sqrt, Yeo-Johnson, min-max / z-score / robust scaling, binning) |
| Data encoding | NB 3.7 (label, ordinal, one-hot, frequency, leak-safe target encoding) |

### Unit IV - Python foundations ([unit4_python_foundations.ipynb](unit4_python_foundations.ipynb))
| Syllabus topic | Where |
|---|---|
| NumPy arrays and vectors, indexing, slicing, multi-dimensional arrays, reshaping, vectorised operations | NB 4.1 (including broadcasting and a loop-vs-vectorised timing) |
| Matrices, matrix subsetting `array[row_slice, col_slice]` | NB 4.2 (determinant, inverse, solve, eigen; correlation matrix by matrix product) |
| PCA | NB 4.3 (from first principles with `eigh`, checked against scikit-learn; scree plot, loadings); runs the project's previously unused `detect_feature_corruption_pca` |
| pandas `loc` / `iloc`, categorical data (levels, summaries), extending DataFrames, sorting | NB 4.4 |
| Lists and dictionaries; analysing a CSV with lists and dictionaries | NB 4.5 (`csv.DictReader`, `defaultdict`, `Counter`, verified against pandas) |

### Unit V - EDA, feature engineering, evaluation ([unit5_eda_features_evaluation.ipynb](unit5_eda_features_evaluation.ipynb))
| Syllabus topic | Where |
|---|---|
| Importing, `head` / `tail` / `info` / `describe` | NB 5.1 |
| Aggregation and grouping | NB 5.2 (`agg`, `transform`, `filter`, `pivot_table`, `crosstab`); TC `phase7_impact.py` |
| Merging and concatenating DataFrames | NB 5.3 (inner/left/right/outer, `validate`, `indicator`, `concat`) |
| EDA: descriptive statistics, distribution analysis, correlation analysis, trends and patterns | NB 5.4 (Pearson, Spearman, Cramer's V; trend plots) |
| Feature extraction, selection, scaling | NB 5.5 (engineered features, filter / embedded selection, effect of scaling by model) |
| Train / validation / test split, cross-validation | NB 5.6; TC `data.three_way_split`, out-of-fold detectors in `detect.py` |
| Confusion matrix, accuracy, precision, recall | NB 5.7 (by hand and with scikit-learn, threshold trade-off, ROC, reproduces the Phase-2 baseline); TC `baseline.evaluate` |

### Unit VI - Visualization and regression ([unit6_visualization_regression.ipynb](unit6_visualization_regression.ipynb))
| Syllabus topic | Where |
|---|---|
| Chart anatomy, legend | NB 6.1 |
| Bar chart (grouped, stacked) | NB 6.2 |
| Histogram | NB 6.3 |
| Box plot (and violin) | NB 6.4 |
| Line graph, multiple lines in a line graph | NB 6.5 (single line, twin axes, three lines with distinct styles) |
| Scatter | NB 6.6 (over-plotting, hexbin, regression line, scatter-plot matrix) |
| 2D and 3D visualization | NB 6.7 (heatmap, contours, 3-D scatter, surface, bars) |
| Core visualizations with Matplotlib and Seaborn | NB 6.1-6.8 |
| Linear regression analysis | NB 6.9 (normal equation by hand vs scikit-learn, R2, RMSE) |
| Multiple linear regression | NB 6.10-6.12 (OLS with standard errors, t / p / CI, adjusted R2, F-test, standardised coefficients, residual diagnostics, VIF, one-time test evaluation) |

## Two things worth knowing

- **A plotting pitfall found while building these.** In this environment (pandas 3.0 with seaborn 0.13.2) the figure-level
  `sns.catplot(kind="bar", col=...)` drew bar heights that did **not** match a plain `groupby`. The notebook draws those panels
  with the axes-level `sns.barplot` instead and prints a `groupby` check next to the chart. Axes-level `barplot`, and
  `FacetGrid.map(histplot)`, were checked against ground truth and agree.
- **Regression is new material, not a change to the research.** TaskClean is a classification project, so Unit VI regression
  predicts the numeric attribute `hours-per-week`. The final section shows why a 0/1 target needs logistic regression instead.
