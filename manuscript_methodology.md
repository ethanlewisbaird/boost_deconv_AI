# Methods

## Overview
The ML_AI Boost Deconvolution package is a computational framework developed to address the challenges of deconvoluting bulk and single-cell RNA sequencing (RNA-seq) data. By leveraging advanced machine learning algorithms, the package enables accurate estimation of cell type proportions and provides a comprehensive suite of tools for data preprocessing, model training, evaluation, and visualization. This methodology is particularly suited for transcriptomic studies aiming to decipher cellular heterogeneity in complex biological samples.

## Data Input
The ML_AI Boost Deconvolution package supports two primary types of input data:
- **Bulk RNA-seq data**: Gene expression matrices derived from bulk RNA sequencing experiments, typically provided as raw counts or normalized values such as Transcripts Per Million (TPM).
- **Single-cell RNA-seq data**: High-dimensional gene expression data from single-cell sequencing experiments. The package supports various formats and includes preprocessing tools to ensure data quality.

## Data Preprocessing
To ensure high-quality input for downstream analysis, the package implements a robust preprocessing pipeline:
- **Gene Filtering**: Lowly expressed genes are removed based on user-defined thresholds for minimum expression levels and the number of cells expressing a gene. This step reduces noise and enhances the signal-to-noise ratio.
- **Normalization**: The package supports multiple normalization techniques, including log-transformation ($x' = \log_2(x + 1)$), Z-score normalization ($z = \frac{x - \mu}{\sigma}$), and min-max scaling ($x' = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$). These methods correct for technical variability and ensure comparability across samples.
- **Feature Selection**: Informative genes are selected based on statistical measures such as variance ($\text{Var}(x) > \theta$) and correlation ($|\text{Corr}(x, y)| > \rho$), where $\theta$ and $\rho$ are user-defined thresholds.

## Pseudobulk Generation
For single-cell RNA-seq data, the package provides advanced pseudobulk generation methods. These methods aggregate single-cell data into pseudobulk profiles by summing or averaging expression values across cells within predefined groups (e.g., cell types or clusters). This step reduces the complexity of single-cell data while preserving key biological signals.

The pseudobulk aggregation is performed using the following mathematical functions:
- **Sum**: $x_{\text{pseudobulk}} = \sum_{i=1}^n x_i$, where $x_i$ is the expression value for cell $i$.
- **Mean**: $x_{\text{pseudobulk}} = \frac{1}{n} \sum_{i=1}^n x_i$, where $n$ is the number of cells in the group.

## Machine Learning Models
The core of the deconvolution process is based on machine learning models, particularly XGBoost, a gradient boosting framework optimized for high-dimensional data. The following mathematical principles underpin the model:
- **Gradient Boosting**: The model minimizes a loss function $L(\theta)$ by iteratively adding weak learners $h_m(x)$:
  $$F_m(x) = F_{m-1}(x) + \eta h_m(x)$$
  where $F_m(x)$ is the ensemble model at iteration $m$, $\eta$ is the learning rate, and $h_m(x)$ is the weak learner.
- **Objective Function**: The objective function combines a loss function $l$ and a regularization term $\Omega$:
  $$\text{Obj}(\theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{j=1}^T \Omega(h_j)$$
  where $y_i$ is the true label, $\hat{y}_i$ is the predicted label, and $T$ is the number of trees.
- **Regularization**: The regularization term $\Omega$ is defined as:
  $$\Omega(h) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
  where $\gamma$ and $\lambda$ are regularization parameters, and $w_j$ are the weights of the tree nodes.

The package also includes custom calibration methods to improve the reliability of probability estimates, ensuring that the predicted cell type proportions are interpretable and biologically meaningful. Cross-validation is employed to evaluate model performance and prevent overfitting, with customizable parameters such as the number of folds.

## Prediction
The trained models are used to predict cell type proportions in bulk RNA-seq data. The package provides separate scripts for predicting bulk and pseudobulk data. Predictions are output as CSV files containing:
- **Predicted Cell Type Proportions**: The relative abundance of each cell type in each sample.
- **Prediction Probabilities**: Confidence scores for each predicted cell type proportion.

## Evaluation
The performance of the deconvolution models is rigorously evaluated using a suite of metrics:
- **Accuracy**: $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Precision**: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall**: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
- **F1-Score**: $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Area Under the ROC Curve (AUC)**: Measures the ability of the model to distinguish between classes, calculated as the area under the ROC curve.

Evaluation is performed at multiple levels:
- **Global Metrics**: Overall performance metrics across all samples.
- **Per-Sample Metrics**: Performance metrics for individual samples, allowing users to identify samples with suboptimal predictions.
- **No-Microscopy Evaluation**: Evaluation of models without microscopy data, enabling users to assess the robustness of the models in the absence of additional experimental data.

## Visualization
The package includes a suite of visualization tools to facilitate data interpretation and presentation. These tools generate publication-quality figures, including:
- **Bulk Deconvolution Plots**: Bar plots and heatmaps illustrating deconvolution results for bulk RNA-seq data.
- **Prediction Plots**: Scatter plots and density plots visualizing predicted cell type proportions.
- **Evaluation Plots**: Graphical summaries of model performance metrics, such as ROC curves and precision-recall curves.

## Implementation and Availability
The ML_AI Boost Deconvolution package is implemented in Python and relies on widely used libraries such as NumPy, pandas, scikit-learn, and XGBoost. The package is open-source and available for download, with detailed installation and usage instructions provided in the accompanying documentation.

To install the package, users can execute the `install_and_extract_ML_AI_boost_deconv.sh` script, which sets up the required environment and installs all dependencies listed in the `requirements.txt` file. The pipeline can be configured via the `config.yaml` file, which allows users to specify input data paths, model parameters, and output directories. The entire pipeline can be executed using the `run_pipeline.py` script, which orchestrates all steps from data preprocessing to model training, prediction, and evaluation.

## Conclusion
The ML_AI Boost Deconvolution package represents a significant advancement in the field of transcriptomics, providing a robust and user-friendly solution for deconvolution of RNA-seq data. By integrating advanced machine learning techniques with comprehensive preprocessing, evaluation, and visualization tools, the package enables researchers to gain deeper insights into cellular heterogeneity and underlying biological processes.

# Methods and Algorithms of the ML_AI Boost Deconvolution Package

## Overview
The ML_AI Boost Deconvolution package is a comprehensive tool designed for the deconvolution of bulk and single-cell RNA sequencing data. It leverages machine learning techniques to provide accurate and efficient deconvolution of complex biological datasets, enabling researchers to extract meaningful insights from high-dimensional data. The package is particularly suited for applications in transcriptomics, where understanding the cellular composition of bulk RNA-seq samples is critical for interpreting biological processes and disease mechanisms.

## Key Features
1. **Data Preprocessing**: Robust pipelines for filtering and normalizing gene expression data, supporting multiple input formats and handling missing data and outliers.
2. **Pseudobulk Generation**: Advanced methods to aggregate single-cell data for improved signal-to-noise ratio.
3. **Machine Learning Models**: State-of-the-art models, including XGBoost, optimized for high-dimensional data.
4. **Evaluation Metrics**: Comprehensive metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
5. **Visualization Tools**: Scripts for generating publication-quality visualizations, including heatmaps, bar plots, and scatter plots.

---

## Detailed Methodology

### 1. Data Input
The package supports:
- **Bulk RNA-seq data**: Count matrices or TPM values.
- **Single-cell RNA-seq data**: High-dimensional data with preprocessing and quality control tools.

### 2. Preprocessing
- **Gene Filtering**: Removes lowly expressed genes based on user-defined thresholds.
- **Normalization**: Includes log-transformation, scaling, and batch effect correction.
- **Feature Selection**: Identifies informative genes using variance, correlation, or biological relevance.

### 3. Pseudobulk Generation
- Aggregates single-cell data into pseudobulk profiles by summing or averaging expression values across predefined groups (e.g., cell types or clusters).
- Reduces technical noise and sparsity in single-cell data.

### 4. Model Training
- **XGBoost**: A gradient boosting framework optimized for sparse, high-dimensional data. It uses decision trees as base learners and minimizes a regularized objective function to improve predictive accuracy.
- **Custom Calibration**: Adjusts model outputs to ensure biologically meaningful predictions.
- **Cross-Validation**: Implements k-fold cross-validation to evaluate model performance and prevent overfitting.

### 5. Prediction
- Predicts cell type proportions in bulk RNA-seq data.
- Outputs include predicted cell type proportions and confidence scores.

### 6. Evaluation
The performance of the deconvolution models is evaluated using:
- **Global Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC.
- **Per-Sample Metrics**: Identifies samples with suboptimal predictions.
- **No-Microscopy Evaluation**: Assesses model robustness without additional experimental data.
- **Custom Metrics**: Users can define and implement custom evaluation metrics to suit their specific research needs.

### 7. Visualization
The package includes scripts for generating visualizations, such as:
- **Bulk Deconvolution Plots**: Bar plots and heatmaps of deconvolution results.
- **Prediction Plots**: Scatter and density plots of predicted cell type proportions.

---

## Mathematical Functions and Algorithms

#### XGBoost
The XGBoost algorithm is a scalable and efficient implementation of gradient boosting. It minimizes the following regularized objective function:
```
Obj(θ) = Σ[l(yᵢ, ŷᵢ) + ΣΩ(fₖ)]
```
Where:
- `l` is the loss function (e.g., mean squared error for regression or log loss for classification).
- `Ω(fₖ)` is the regularization term to prevent overfitting.

#### Calibration
The package uses isotonic regression or Platt scaling to calibrate the predicted probabilities, ensuring that they are well-calibrated and interpretable.

#### Pseudobulk Aggregation
For single-cell data, pseudobulk profiles are generated by summing or averaging gene expression values across cells within predefined groups:
```
Pseudobulk_g = (1/N) Σ Expression_g,i
```
Where:
- `g` is the gene.
- `N` is the number of cells in the group.

#### Evaluation Metrics
- **Accuracy**: 
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Precision**: 
```
Precision = TP / (TP + FP)
```
- **Recall**: 
```
Recall = TP / (TP + FN)
```
- **F1-Score**: 
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```