# AI-Driven Solutions for Sustainability
## Executive Summary
Three end-to-end deep learning case studies were completed under a simulated multi-client consulting engagement, each tackling a different data modality and business problem.

**Solar Power Forecasting (Tabular Regression):** Linear regression and 15 MLP variants were compared, with the best model, Dense (200-100-50) with 0.3 Dropout, achieving a correlation of 0.883 and MAE of 317.82 kW.

**Recyclable Material Classification (Image Classification):** Multiple CNN designs were evaluated across 2,691 images in six categories, with the top model, a double-convolution CNN with 0.5 Dropout, reaching 74.75% test accuracy.

**PM2.5 Air-Quality Forecasting (Multivariate Time Series):** 15 RNN variants were tested across ~35,000 hourly records, with a GRU-100 and Dense-24 output achieving the best result at a test MAE of 48.67.

Across all three projects, simpler, well-regularised architectures consistently outperformed deeper or hybrid alternatives, confirming that data quality, regularisation, and inductive bias are the binding constraints, not model capacity.
## Business Problem
The engagement was structured as three independent client requests delivered through MindMesh Innovations, a simulated consultancy.

**Solar Power Forecasting (Renewable Energy):** Photovoltaic output varies with weather and panel geometry, causing grid instability and revenue loss. The task was to predict generated power (kW) from weather variables and solar-geometry parameters (zenith, azimuth, angle of incidence).

**Recyclable Material Classification (Waste Management):** Manual sorting is slow and error-prone, with misclassification contaminating downstream material streams. The task was to classify images into six categories (cardboard, clothes, green glass, white glass, metal, paper) to support automated sorting.

**PM2.5 Air-Quality Forecasting (Public Health):** Fine particulate matter is a leading pollutant linked to respiratory illness and increased hospital admissions. The task was to forecast hourly PM2.5 concentration 24 hours ahead from a multivariate sensor stream covering 2013–2017, to support early warnings and policy decisions.
## Methodology
A consistent workflow was applied across all three notebooks, adapted to each data modality.
#### 1. Data exploration & cleaning
Distribution checks, missing-value handling (imputation for Project 3), feature/target separation.
#### 2. Modality-specific pre-processing
- Project 1: normalisation of 20 continuous predictors so high-magnitude variables (e.g. irradiance) do not dominate gradient updates.
- Project 2: image resizing to 100×100×3, pixel normalisation, one-hot label encoding.
- Project 3: missing-value imputation, scaling, and windowing into 50-hour input sequences with a 24-hour forecast horizon.
#### 3. Train/test split
- Projects 1 & 2: 70/30 random split.
- Project 3: temporal split; all data before 2016 used for training, January 2016 used as test set to prevent look-ahead leakage.
#### 4. Baselines before deep learning
- Linear Regression in Project 1.
- Baseline MLP before CNN variants in Project 2.
- Basic RNN before LSTM, GRU, bidirectional, and hybrid models in Project 3.
#### 5. Systematic experimentation
16 model variants in Project 1, multiple CNN architectures in Project 2, 15 RNN variants in Project 3. Every run logged with hyperparameters and metrics.
#### 6. Reproducibility
All experiments use fixed random seeds and deterministic-operation flags (TF_DETERMINISTIC_OPS, PYTHONHASHSEED, tf.random.set_seed).
## Demonstration of Skills and Capabilities
**Deep Learning Architectures:** MLPs with varying depth, width, and dropout; CNNs with single- and double-convolution blocks across varied kernel sizes and filter counts; and RNN variants including vanilla RNN, LSTM, GRU, bidirectional, stacked, and CNN-RNN hybrids.

**Model Evaluation:** Regression tasks used MAE, RMSE, and correlation with scatter plots and loss curves; classification used accuracy, confusion matrices, and per-class error analysis; time series used MAE alongside qualitative trend inspection across 10,000+ test hours.

**Engineering Practice:** Reproducible Keras pipelines with seeded RNGs, EarlyStopping, dropout regularisation, appropriate optimiser selection (RMSprop, Nadam), and leakage-free sliding-window construction for sequence models.

**Business Translation:** Model metrics were linked to client outcomes (grid balancing, sorting throughput, public-health warnings), class-level failure modes were identified with root-cause analysis, and deployment requirements were articulated across hardware integration, retraining cadence, and uncertainty estimation.
## Results and Business Recommendations
**Project 1 — Solar Power Forecasting:** The best model, Dense-200-100-50 with Dropout(0.3), achieved a correlation of 0.883 and MAE of 317.82 kW, outperforming the linear regression baseline (correlation 0.707, MAE 390.60 kW) by capturing non-linear weather and solar geometry interactions. For deployment, integrate short-horizon weather feeds with periodic retraining for seasonal drift, and add uncertainty estimates before using outputs for spot-market decisions.

**Project 2 — Recyclable Material Classification:** The best model, a double-convolution CNN (30→60→90 filters) with a 120-unit dense head and Dropout(0.5), reached 74.75% test accuracy. Green glass was easiest to classify due to distinct visual features, while white glass, paper, and metal were frequently confused due to similar reflectance and texture. Dataset scale is the binding constraint; expanding training data, applying augmentation, and adopting transfer learning (e.g. ResNet, EfficientNet) are recommended before deployment.

**Project 3 — PM2.5 Forecasting:** The best model, GRU-100 + Dense-24, achieved a test MAE of 48.67. Simpler architectures consistently outperformed stacked and hybrid alternatives, as added capacity increased overfitting. Predictions tracked overall trends well but smoothed sharp pollution spikes. Before driving public-health alerts, pair forecasts with probabilistic prediction intervals, exogenous features (traffic, emissions, satellite indices), and scheduled retraining to handle concept drift.
## Model Integrity & Risk Assessment
**Reproducibility:** All notebooks use fixed seeds with experiment tables reporting metrics for every architecture trialled, enabling end-to-end audit of modelling decisions.

**Generalisation Gap:**
- Project 2 shows a notable gap (89.5% train vs 74.75% test), indicating sensitivity to out-of-distribution image conditions.
- Projects 1 and 3 show smaller gaps with stable validation curves.
**Limitations:**

- Project 2: 2,691 images is small for six-class image classification; transfer learning from a pretrained backbone (e.g. ResNet, EfficientNet) was not applied and would likely push accuracy above 90%. The dataset, not the architecture, is the constraint.
- Project 3: A single one-month test window (January 2016) limits confidence in the headline MAE; a walk-forward evaluation across seasons would be more robust.

**Deployment Risks Beyond Modelling:**
- Data-pipeline reliability (sensor downtime, image quality degradation).
- Concept drift requiring scheduled retraining.
## Context and Credits
Client: MindMesh Innovations multi-client engagement is a scenario provided in the assessment specification.

Tools Used: Python 3, TensorFlow / Keras, scikit-learn, pandas, NumPy, Matplotlib, Jupyter, Google Colab (GPU runtime).

Analyst: Quang Huy Le



