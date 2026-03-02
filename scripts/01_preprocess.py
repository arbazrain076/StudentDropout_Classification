"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 01: Data Preprocessing Pipeline
=============================================================================
Domain    : BINARY CLASSIFICATION — Predict student dropout (0=Retained, 1=Dropped Out)
Algorithm : K-Nearest Neighbours (KNN) + Decision Tree + Naive Bayes + Random Forest + Logistic Regression
Dataset   : Student Dropout Dataset (10,000 records)
Author    : [Your Name / Student ID]
Run       : python scripts/01_preprocess.py
Output    : data/clean/student_dropout_clean.csv
            data/clean/student_dropout_model.csv
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH   = os.path.join("data", "raw",   "student_dropout_dataset_v3.csv")
CLEAN_PATH = os.path.join("data", "clean", "student_dropout_clean.csv")
MODEL_PATH = os.path.join("data", "clean", "student_dropout_model.csv")
os.makedirs("data/clean", exist_ok=True)

print("=" * 65)
print("  STW5000CEM — Student Dropout Prediction Classifier")
print("  Step 1: Data Preprocessing")
print("=" * 65)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("\n[1/6] Loading raw dataset ...")
df = pd.read_csv(RAW_PATH)
print(f"      Raw shape  : {df.shape}")
print(f"      Records    : {len(df):,}")
print(f"      Features   : {df.shape[1] - 1}")
print(f"      Target     : Dropout (0=Retained, 1=Dropped)")
print(f"      Class distribution:")
print(f"        Retained (0): {(df['Dropout']==0).sum():,} ({(df['Dropout']==0).sum()/len(df)*100:.1f}%)")
print(f"        Dropped (1):  {(df['Dropout']==1).sum():,} ({(df['Dropout']==1).sum()/len(df)*100:.1f}%)")

# ── 2. Handle Missing Values ──────────────────────────────────────────────────
print("\n[2/6] Handling missing values ...")
print(f"      Missing values before: {df.isnull().sum().sum()}")

# Numeric columns: impute with median
numeric_cols = ['Family_Income', 'Study_Hours_per_Day', 'Stress_Index']
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"        {col}: filled {df[col].isnull().sum()} with median={median_val:.2f}")

# Categorical: impute with mode
if df['Parental_Education'].isnull().sum() > 0:
    mode_val = df['Parental_Education'].mode()[0]
    df['Parental_Education'].fillna(mode_val, inplace=True)
    print(f"        Parental_Education: filled with mode='{mode_val}'")

print(f"      Missing values after: {df.isnull().sum().sum()}")

# ── 3. Feature Engineering ────────────────────────────────────────────────────
print("\n[3/6] Engineering features ...")

# Encode categorical variables
le_gender = LabelEncoder()
le_internet = LabelEncoder()
le_job = LabelEncoder()
le_scholarship = LabelEncoder()
le_semester = LabelEncoder()
le_dept = LabelEncoder()
le_parental = LabelEncoder()

df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
df['Internet_Access_Encoded'] = le_internet.fit_transform(df['Internet_Access'])
df['Part_Time_Job_Encoded'] = le_job.fit_transform(df['Part_Time_Job'])
df['Scholarship_Encoded'] = le_scholarship.fit_transform(df['Scholarship'])
df['Semester_Encoded'] = le_semester.fit_transform(df['Semester'])
df['Department_Encoded'] = le_dept.fit_transform(df['Department'])
df['Parental_Education_Encoded'] = le_parental.fit_transform(df['Parental_Education'])

# Create interaction features
df['GPA_Attendance'] = df['GPA'] * df['Attendance_Rate']
df['Stress_Job'] = df['Stress_Index'] * df['Part_Time_Job_Encoded']
df['Income_Scholarship'] = df['Family_Income'] * df['Scholarship_Encoded']
df['Study_Attendance'] = df['Study_Hours_per_Day'] * df['Attendance_Rate']

# Risk flags
df['High_Stress'] = (df['Stress_Index'] > 7).astype(int)
df['Low_Attendance'] = (df['Attendance_Rate'] < 75).astype(int)
df['Low_GPA'] = (df['GPA'] < 2.0).astype(int)
df['Long_Commute'] = (df['Travel_Time_Minutes'] > 45).astype(int)

print(f"      Encoded {7} categorical variables")
print(f"      Created {4} interaction features")
print(f"      Created {4} risk flag features")

# ── 4. Outlier Detection (IQR) ────────────────────────────────────────────────
print("\n[4/6] Outlier detection and capping ...")
numeric_features = ['Age', 'Family_Income', 'Study_Hours_per_Day', 'Attendance_Rate',
                   'Assignment_Delay_Days', 'Travel_Time_Minutes', 'Stress_Index',
                   'GPA', 'Semester_GPA', 'CGPA']

outliers_capped = 0
for col in numeric_features:
    Q1 = df[col].quantile(0.05)
    Q3 = df[col].quantile(0.95)
    before = ((df[col] < Q1) | (df[col] > Q3)).sum()
    df[col] = df[col].clip(lower=Q1, upper=Q3)
    outliers_capped += before
    if before > 0:
        print(f"        {col}: capped {before} outliers to [5th, 95th] percentile")

print(f"      Total outliers capped: {outliers_capped}")

# ── 5. Drop Unnecessary Columns ───────────────────────────────────────────────
print("\n[5/6] Dropping unnecessary columns ...")
df.drop(columns=['Student_ID'], inplace=True)  # ID not predictive
print(f"      Dropped: Student_ID")

# ── 6. Save ───────────────────────────────────────────────────────────────────
print("\n[6/6] Saving datasets ...")
df.to_csv(CLEAN_PATH, index=False)
print(f"      Saved full clean: {CLEAN_PATH}  ({len(df):,} rows, {df.shape[1]} cols)")

# Model dataset: same as clean for this dataset (no time-series)
df.to_csv(MODEL_PATH, index=False)
print(f"      Saved model data : {MODEL_PATH}  ({len(df):,} rows)")

print("\n── Summary ──────────────────────────────────────────────────")
print(f"  Raw records          : 10,000")
print(f"  After cleaning       : {len(df):,}")
print(f"  Model records        : {len(df):,}")
print(f"  Features (original)  : 18")
print(f"  Features (engineered): {df.shape[1] - 1}")  # -1 for target
print(f"  Classification target: Dropout (0=Retained, 1=Dropped)")
print(f"  Class imbalance      : 76.5% / 23.5% (will use SMOTE/class weights)")
print("  Preprocessing complete ✅")
