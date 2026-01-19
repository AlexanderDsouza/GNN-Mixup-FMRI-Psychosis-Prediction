# GNN Mixup on Brain fMRI for Psychosis Prediction

This repository contains code and data for a graph neural network (GNN) + mixup approach applied to **brain fMRI scans** for predicting psychosis. The project integrates fMRI graph data and demographic information to improve patient-level predictions using graph embeddings and data augmentation.  

---

## Dataset

- **Patients:** 56 individuals  
- **Data types:**
  - Brain fMRI graphs for each patient
  - Demographic data for binary classification (psychosis vs. control)
- **Files:**
  - `.xlsx` files: Raw brain graph data
  - `.csv` file: Spectral embeddings of all patient graphs

---

## Project Workflow

1. **Preprocessing & Graph Construction**  
   - Graphs created from fMRI scans using **absolute value adjacency matrices**
   - 56 patient brain scan graphs prepared for downstream modeling

2. **Graph Embedding & Mixup**  
   - Spectral embeddings computed (symmetric normalized Laplacian)
   - **Mixup augmentation**:
     - Synthetic instances generated via interpolation of embeddings
     - Categorical demographics: probabilistic sampling
     - Continuous demographics: linear interpolation

3. **Patient-to-Patient Graph**  
   - All instances connected based on demographic similarity
   - Spectral embeddings updated using a linear combination weighted by connected patient similarity

4. **Model Training**  
   - Train GNN on the augmented patient graph
   - Evaluate performance with cross-validation metrics
   - Create graph summaries and train simple models like Logistic Regression, SVR, and MLPRegressor.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `GNN.ipynb` | Full architecture workflow: preprocess fMRI data, create spectral embeddings, apply mixup, build patient similarity graph, and train + test GNN. |
| `CorrelationToOutcomes.ipynb` | Compute graph summary statistics (strength, clustering) and correlations with outcomes. Implements logistic regression with intra- and inter-class mixup. Includes 20-run 5-fold CV metrics. |
| `LogisticRegressionMixup.ipynb` | Logistic regression, MLP, and SVR models on graph summary features with mixup. |
| `BaselineGNN.ipynb` | Basic feedforward neural network (FFN) on patient data as a baseline. |
| `Including_bprs_baseline_LogisticRegressionMixup.ipynb` | Variant of logistic regression mixup including the key demographic feature BPRS baseline. |
| `ConnectivityAsFeaturesAnalysis.ipynb` | Uses connectivity features of brain areas instead of graph structure for modeling. |
| `Atlas_rest_results.ipynb` | Exploratory analysis to understand the fMRI dataset. |

---

## Metrics

Examples of cross-validation performance from `CorrelationToOutcomes.ipynb`:

| Model | Accuracy | Precision | Recall | F1-score | AUC |
|-------|---------|-----------|--------|----------|-----|
| Logistic Regression (intra-mixup) | 0.620 | 0.649 | 0.605 | 0.610 | 0.727 |
| Linear Regression (inter-mixup) | 0.639 | 0.669 | 0.595 | 0.610 | 0.734 |
| Logistic Regression (no mixup) | 0.634 | 0.657 | 0.647 | 0.636 | 0.740 |

---

## Usage

1. Clone the repo:

```bash
git clone https://github.com/yourusername/gnn-mixup-fmri-psychosis-prediction.git
cd gnn-mixup-fmri-psychosis-prediction
