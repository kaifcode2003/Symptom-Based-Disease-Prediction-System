# 🩺 Symptom-Based Disease Prediction System  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)  
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-lightgrey.svg)  
![Accuracy](https://img.shields.io/badge/Accuracy-97.6%25-brightgreen.svg)  
![F1 Score](https://img.shields.io/badge/F1--Macro-0.98-green.svg)  
![License](https://img.shields.io/badge/License-MIT-blue.svg)  

---

## 📌 Project Overview  
This project implements a **machine learning–based disease prediction system** that classifies patient illnesses from **132 symptoms** into one of **41 possible diseases**.  
Using a **RandomForestClassifier** with hyperparameter tuning and rigorous evaluation, the model achieved an impressive **97.6% test accuracy** and **0.98 F1-macro score**.  

The workflow demonstrates a **complete ML pipeline**, from data exploration and preprocessing to model training, optimization, and final submission file generation.  

---

## ✨ Features  

- 🔍 **Comprehensive Data Analysis**: Exploratory Data Analysis (EDA) for feature distribution and class balance.  
- 🌲 **Robust Model Training**: RandomForestClassifier ensures strong generalization across diseases.  
- 🎯 **High Predictive Accuracy**: 97.62% test accuracy and 0.98 F1-macro score.  
- ⚡ **Hyperparameter Optimization**: GridSearchCV with 5-fold stratified cross-validation.  
- 📊 **Performance Visualization**: Confusion matrix and plots for evaluation insights.  
- 📂 **End-to-End Workflow**: From data loading to generating `submission.csv`.  

---

## 📂 Dataset  

The dataset consists of **two CSV files**:  

- **`Training.csv`** → 4920 records used for training & validation.  
- **`Testing.csv`** → 42 records used for final predictions.  

Each record contains:  
- **132 binary features** (`0` or `1`) representing symptoms (e.g., *itching, skin_rash, chills*).  
- **1 target column**: `prognosis` (disease label).  

✅ The training data is **perfectly balanced**, ensuring fair representation across all 41 diseases.  

---

## ⚙️ Methodology  

1. **📥 Data Loading & Exploration**  
   - Used Pandas to load datasets.  
   - Basic exploration with `.head()`, `.describe()`, `.shape`.  

2. **🧹 Data Cleaning & Preprocessing**  
   - Removed empty column (`Unnamed: 133`).  
   - Converted categorical target (`prognosis`) into numerical labels using `LabelEncoder`.  

3. **📊 Data Visualization**  
   - Visualized target class distribution with Matplotlib & Seaborn.  
   - Confirmed dataset balance across diseases.  

4. **🤖 Model Selection & Training**  
   - Chose **RandomForestClassifier** for classification.  
   - Stratified **70:30 train-validation split**.  

5. **⚡ Hyperparameter Tuning**  
   - Used **GridSearchCV** with 5-fold stratified CV.  
   - Optimized `n_estimators`, `max_depth`, and `min_samples_split`.  

6. **📈 Model Evaluation**  
   - Achieved **100% validation accuracy**.  
   - Confusion matrix confirmed perfect classification on validation data.  

7. **📤 Final Prediction**  
   - Model used to predict on `Testing.csv`.  
   - Generated `submission.csv` as final output.  

---

## 🏆 Results  

| Metric              | Validation Set | Test Set   |  
|----------------------|---------------|------------|  
| **Accuracy**         | 100%          | 97.62%     |  
| **F1-Macro Score**   | 100%          | 98.37%     |  

✅ Results show **excellent generalization** and strong predictive performance.  

---

## 🛠️ Technologies Used  

- **Language:** Python 3  
- **Libraries:**  
  - `scikit-learn` → RandomForestClassifier, LabelEncoder, GridSearchCV  
  - `pandas` → Data handling  
  - `numpy` → Numerical operations  
  - `matplotlib`, `seaborn` → Visualization  
- **Environment:** Jupyter Notebook  

---

## 🚀 How to Run  
 **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/disease-prediction.git
   cd disease-prediction


