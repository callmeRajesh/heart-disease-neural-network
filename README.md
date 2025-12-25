# Heart Disease Prediction using Neural Networks

This project demonstrates a **neural network model** built using **TensorFlow and Keras** to predict heart disease based on clinical features. It was created as part of my learning journey from traditional machine learning models to deep learning models.

---

## Dataset

- Dataset: [Heart Disease Dataset from Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)  
- The dataset contains 14 columns (features and target) including:
  - `age`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `fbs` (fasting blood sugar), `restecg` (resting ECG results), `thalach` (max heart rate), `exang` (exercise-induced angina), `oldpeak`, `slope`, `ca`, `thal`, and `target`.
- Target variable:
  - `0` → No heart disease
  - `1` → Presence of heart disease

> Note: Non-informative features like `sex` were dropped to simplify the model.

---

## Preprocessing

- Checked for missing values and data types
- Dropped non-informative columns
- Scaled features using **MinMaxScaler** from scikit-learn
- Split dataset into **training** and **testing** sets (80:20)
- Created **validation split** during training

---

## Model

- Framework: **TensorFlow & Keras**
- Architecture:
  - Input layer: number of neurons = number of features
  - Hidden layer: 8 neurons, ReLU activation
  - Output layer: 1 neuron, Sigmoid activation (binary classification)
- Loss: `binary_crossentropy`
- Optimizer: `Adam`
- Metrics: `accuracy`

---

## Training

- Epochs: 100  
- Batch size: 32  
- Validation split: 0.2  

The model was trained and monitored using **training and validation loss/accuracy** to ensure it generalizes well.

---

## Evaluation

- **Test Accuracy**: ~82%  
- **Confusion Matrix** used to analyze false positives and false negatives  

Example confusion matrix:

| Actual\Predicted | 0 | 1 |
|-----------------|---|---|
| 0               | TN| FP|
| 1               | FN| TP|

- **Precision, Recall, F1-score** were also calculated to measure model performance.  
- Observed that **recall for heart disease class (1)** is critical to minimize false negatives.

---

## Key Learnings

- Practiced **TensorFlow & Keras** for structured data
- Learned to move from traditional ML to neural networks
- Learned to properly evaluate binary classification models using **confusion matrix, precision, recall**
- Understood importance of **data preprocessing and scaling** in neural networks
- Built a workflow that can be adapted to **real company datasets**
