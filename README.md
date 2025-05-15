# 🧠 Stroke Prediction with Interpretable Machine Learning

This project applies interpretable machine learning techniques to predict stroke risk using electronic health record (EHR) data. It combines model training, performance evaluation, and explainability techniques (LIME, SHAP, InterpretML) to ensure both predictive accuracy and transparency — crucial for high-stakes domains like healthcare.

---

## 📁 Dataset

- **Source:** [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Features:**
  - Demographics: `gender`, `age`, `ever_married`, etc.
  - Lifestyle/Health: `smoking_status`, `hypertension`, `heart_disease`, `bmi`, etc.
  - Target: `stroke` (binary)

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Missing values in `bmi` filled with zero.
- One-hot encoding for categorical variables.
- Dropped `id` (non-predictive).
- Ensured all features are numeric.

### 2. Class Imbalance Handling
- Applied `RandomOverSampler` to address severe class imbalance (stroke being the minority class).

### 3. Model Training & Evaluation
- Trained multiple interpretable models using [`InterpretML`](https://github.com/interpretml/interpret):
  - Logistic Regression (`penalty='l1'`)
  - Classification Tree
- Metrics used:
  - **Macro F1-Score**
  - **Accuracy**

### 4. Model Interpretability

#### ✅ Model-Specific:
- **InterpretML Dashboards:**
  - `explain_global()` → overall feature importance
  - `explain_local()` → feature influence on individual predictions

#### 🔍 Model-Agnostic:
- **LIME**: Local linear approximations for individual predictions
- **SHAP**: Shapley values showing feature contributions (local + global)

---


> 💡 SHAP and LIME revealed `age`, `hypertension`, and `avg_glucose_level` as key stroke predictors, with insights aligning with clinical expectations.

---

## 🧰 Libraries Used

- `pandas`, `scikit-learn`, `matplotlib`
- `imbalanced-learn`
- `interpret` (InterpretML)
- `lime`
- `shap`

---

## 📌 Key Takeaways

- Handled real-world medical data with class imbalance and missing values.
- Delivered **interpretable predictions** using both model-specific and model-agnostic techniques.
- Built explainability into the ML pipeline, ensuring transparency in decision-making.

---


