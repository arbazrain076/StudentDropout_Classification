# StudentDropout_Classification

**Domain:** Binary Classification (Dropout: 0=Retained, 1=Dropped Out)  
**Dataset:** Student Dropout Dataset (10,000 records)  
**Primary Algorithm:** K-Nearest Neighbours (KNN)  
**Comparison Models:** Decision Tree, Naive Bayes, Random Forest, Logistic Regression

---

## 📊 Dataset Overview

| Attribute | Details |
|-----------|---------|
| Total Records | 10,000 students |
| Features | 18 original + 15 engineered = 33 total |
| Target Variable | Dropout (0=Retained 76.5%, 1=Dropped 23.5%) |
| Missing Values | 2,011 (imputed with median/mode) |
| Date Range | Academic years (Year 1-4) |
| Departments | Arts, Engineering, CS, Business, Science |

**Key Features:**
- **Demographics:** Age, Gender, Family_Income, Parental_Education
- **Academic:** GPA, Semester_GPA, CGPA, Attendance_Rate, Assignment_Delay_Days
- **Lifestyle:** Study_Hours_per_Day, Part_Time_Job, Travel_Time_Minutes, Internet_Access
- **Support:** Scholarship, Stress_Index
- **Engineered:** GPA_Attendance, Stress_Job, High_Stress, Low_Attendance, Low_GPA, Long_Commute

---

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|---------|-----------|--------|----------|---------|
| **KNN (k=5)** | **89.2%** | **0.892** | **0.889** | **0.890** | **0.945** |
| Random Forest | 87.8% | 0.878 | 0.875 | 0.876 | 0.941 |
| Logistic Regression | 85.3% | 0.853 | 0.851 | 0.852 | 0.923 |
| Decision Tree | 82.1% | 0.821 | 0.820 | 0.820 | 0.857 |
| Naive Bayes | 78.4% | 0.784 | 0.782 | 0.783 | 0.891 |

**Best Model:** KNN with k=5 (selected via grid search over k ∈ {1,3,5,7,9,11,15})

**Class-Specific Performance (KNN):**
- **Retained (0):** Precision=0.91, Recall=0.96, F1=0.93
- **Dropped (1):** Precision=0.85, Recall=0.71, F1=0.77

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python scripts/06_run_pipeline.py

# 3. Predict dropout risk for a student
python scripts/05_cli.py
```

---

## 📁 Project Structure

```
student_dropout_project/
├── data/
│   ├── raw/student_dropout_dataset_v3.csv    ← Original dataset
│   └── clean/student_dropout_clean.csv       ← Preprocessed data
├── scripts/
│   ├── 01_preprocess.py     ← Data cleaning + feature engineering
│   ├── 02_eda.py            ← Exploratory data analysis (8 charts)
│   ├── 03_train_evaluate.py← Train 5 models + comparative analysis
│   ├── 04_clustering.py     ← K-Means student profiling
│   ├── 05_cli.py            ← Command-line prediction interface
│   └── 06_run_pipeline.py   ← One-command execution
├── model/
│   ├── best_model.pkl       ← Trained KNN model
│   ├── best_scaler.pkl      ← Feature scaler
│   ├── encoders.pkl         ← Label encoders for categorical vars
│   └── metrics.csv          ← Full performance metrics
├── charts/
│   ├── model_comparison.png
│   ├── confusion_matrix_knn.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── [8 EDA charts]
├── outputs/
│   └── StudentDropoutPredictor_App.html     ← Web GUI
├── README.md
├── requirements.txt
└── DOCUMENTATION_GUIDE.md
```

---

## 🧠 Algorithm Justification

**Why KNN for Student Dropout?**
- **Non-parametric:** No assumptions about feature distributions
- **Distance-based:** Students with similar profiles (GPA, attendance, stress) likely share dropout risk
- **Multi-feature:** Naturally handles 33 features without explicit feature interaction modeling
- **Interpretable:** "This student is similar to 5 others who dropped out" is actionable

**Handling Class Imbalance:**
- Used `class_weight='balanced'` in models
- SMOTE (Synthetic Minority Over-sampling) applied during training
- Evaluated with precision/recall on minority class (Dropout=1)

---

## 📈 Key Findings

1. **Top Predictors (Feature Importance):**
   - GPA (0.18)
   - Attendance_Rate (0.15)
   - Stress_Index (0.12)
   - CGPA (0.11)
   - Study_Hours_per_Day (0.09)

2. **High-Risk Profile:**
   - GPA < 2.0
   - Attendance < 75%
   - Stress > 7
   - Part-time job + Long commute (>45 min)
   - No scholarship

3. **Protective Factors:**
   - High family income
   - Internet access
   - Scholarship
   - Parental education (Bachelor+)

---

## 🎯 Real-World Impact

**For Universities:**
- Early warning system (predict dropout before it happens)
- Targeted interventions (counseling, tutoring, financial aid)
- Resource allocation (focus on high-risk students)

**For Students:**
- Self-assessment tool
- Identify specific risk factors to address
- Motivational insights

**For Policymakers:**
- Understand systemic dropout drivers
- Evaluate intervention effectiveness
- Data-driven policy decisions


