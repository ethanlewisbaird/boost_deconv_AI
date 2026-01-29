# Methods

## Overview
The ML_AI Boost Deconvolution package is a computational framework developed to address the challenges of deconvoluting bulk and single-cell RNA sequencing (RNA-seq) data. By leveraging advanced machine learning approaches, the package aims to estimate cell type proportions from bulk data and to provide tools for working with single-cell data.

## Data Input
The ML_AI Boost Deconvolution package supports two primary types of input data:
- **Bulk RNA-seq data**: Gene expression matrices derived from bulk RNA sequencing experiments, typically provided as raw counts or normalized values such as Transcripts Per Million (TPM).
- **Single-cell RNA-seq data**: High-dimensional gene expression data from single-cell sequencing experiments. The package supports various formats and includes preprocessing tools to ensure data quality.

## Data Preprocessing
To ensure high-quality input for downstream analysis, the package implements a robust preprocessing pipeline:
- **Gene Filtering**: Lowly expressed genes are removed based on user-defined thresholds for minimum expression levels and the number of cells expressing a gene.
- **Normalization**: The package supports multiple normalization techniques, including log-transformation $x' = \log_2(x + 1)$, Z-score normalization $z = \frac{x - \mu}{\sigma}$, and min-max scaling.
- **Feature Selection**: Informative genes are selected based on statistical measures such as variance $\mathrm{Var}(x) > \theta$ and correlation $|\mathrm{Corr}(x, y)| > \rho$, where $\theta$ and $\rho$ are user-defined thresholds.

## Pseudobulk Generation
For single-cell RNA-seq data, the package provides advanced pseudobulk generation methods. These methods aggregate single-cell data into pseudobulk profiles by summing or averaging expression values across groups (e.g., per cell type or cluster).

The pseudobulk aggregation is performed using the following mathematical functions:

- Sum (per gene):
  $$
  x_{\text{pseudobulk}} = \sum_{i=1}^n x_i,
  $$
  where $x_i$ is the expression value for cell $i$.

- Mean (per gene):
  $$
  x_{\text{pseudobulk}} = \frac{1}{n} \sum_{i=1}^n x_i,
  $$
  where $n$ is the number of cells in the group.

## Machine Learning Models
The core of the deconvolution process is based on machine learning models, particularly XGBoost, a gradient boosting framework optimized for high-dimensional data.

- Gradient boosting is built by iteratively adding weak learners $h_m(x)$ to the ensemble:
  $$
  F_m(x) = F_{m-1}(x) + \eta\, h_m(x),
  $$
  where $F_m$ is the ensemble at iteration $m$, $\eta$ is the learning rate, and $h_m$ is the weak learner added at step $m$.

- Objective function. XGBoost minimizes a regularized objective combining the loss and a regularization term:
  $$
  \mathrm{Obj}(\theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k),
  $$
  where $l$ is the per-example loss (e.g., mean squared error for regression or log loss for classification), $f_k$ are the individual trees, and $\Omega$ is a regularization function.

- Regularization (tree complexity penalty) is commonly written as:
  $$
  \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2,
  $$
  where $T$ is the number of leaves in the tree, $w_j$ are leaf weights, and $\gamma,\lambda$ are regularization hyperparameters.

The package also includes custom calibration methods (e.g., isotonic regression or Platt scaling) to improve the reliability of probability estimates and make predicted cell type proportions interpretable.

## Prediction
The trained models are used to predict cell type proportions in bulk RNA-seq data. The package provides separate scripts for predicting bulk and pseudobulk data. Predictions are output as CSV files containing:
- **Predicted Cell Type Proportions**: The relative abundance of each cell type in each sample.
- **Prediction Probabilities**: Confidence scores for each predicted cell type proportion.

## Evaluation
The performance of the deconvolution models is rigorously evaluated using a suite of metrics:

- Accuracy:
  $$
  \mathrm{Accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP} + \mathrm{TN} + \mathrm{FP} + \mathrm{FN}}
  $$
- Precision:
  $$
  \mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}
  $$
- Recall:
  $$
  \mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}
  $$
- F1-score:
  $$
  \mathrm{F1} = 2 \cdot \frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
  $$
- Area Under the ROC Curve (AUC): the area under the ROC curve measuring class separability.

Evaluation is performed at multiple levels:
- **Global Metrics**: Overall performance metrics across all samples.
- **Per-Sample Metrics**: Performance metrics for individual samples, allowing users to identify samples with suboptimal predictions.
- **No-Microscopy Evaluation**: Evaluation of models without microscopy data, enabling users to assess robustness in absence of additional experimental data.

## Visualization
The package includes a suite of visualization tools to facilitate data interpretation and presentation. These tools generate publication-quality figures, including:
- **Bulk Deconvolution Plots**: Bar plots and heatmaps illustrating deconvolution results for bulk RNA-seq data.
- **Prediction Plots**: Scatter plots and density plots visualizing predicted cell type proportions.
- **Evaluation Plots**: Graphical summaries of model performance metrics, such as ROC curves and precision–recall curves.

## Implementation and Availability
The ML_AI Boost Deconvolution package is implemented in Python and relies on widely used libraries such as NumPy, pandas, scikit-learn, and XGBoost. The package is open-source and available for download.

To install the package, users can execute the `install_and_extract_ML_AI_boost_deconv.sh` script, which sets up the required environment and installs all dependencies listed in `requirements.txt`.

## Conclusion
The ML_AI Boost Deconvolution package represents a significant advancement in the field of transcriptomics, providing a robust and user-friendly solution for deconvolution of RNA-seq data.