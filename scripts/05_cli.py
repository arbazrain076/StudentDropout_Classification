"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 05: Command-Line Interface (CLI)
=============================================================================
Purpose : Interactive CLI to predict student dropout risk
Run     : python scripts/05_cli.py
          python scripts/05_cli.py --gpa 1.8 --attendance 65 --stress 8
=============================================================================
"""

import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd

MODEL_DIR = "model"

def load_model():
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "best_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    return model, scaler, meta

FEATURE_COLS = [
    'Age', 'Gender_Encoded', 'Family_Income', 'Internet_Access_Encoded',
    'Study_Hours_per_Day', 'Attendance_Rate', 'Assignment_Delay_Days',
    'Travel_Time_Minutes', 'Part_Time_Job_Encoded', 'Scholarship_Encoded',
    'Stress_Index', 'GPA', 'Semester_GPA', 'CGPA', 'Semester_Encoded',
    'Department_Encoded', 'Parental_Education_Encoded', 'GPA_Attendance',
    'Stress_Job', 'Income_Scholarship', 'Study_Attendance', 'High_Stress',
    'Low_Attendance', 'Low_GPA', 'Long_Commute'
]

def predict(features, model, scaler):
    """Predict dropout probability"""
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    pred = model.predict(X_scaled)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        dropout_prob = proba[1]
    else:
        dropout_prob = pred
    
    return pred, dropout_prob

def print_result(pred, prob, features_dict):
    risk_level = "🔴 HIGH RISK" if prob > 0.6 else "🟡 MEDIUM RISK" if prob > 0.4 else "🟢 LOW RISK"
    
    print(f"\n  ┌──────────────────────────────────────────────────")
    print(f"  │  {risk_level}")
    print(f"  │  Dropout Probability: {prob*100:.1f}%")
    print(f"  │  Prediction: {'⚠️ Likely to Drop Out' if pred == 1 else '✅ Likely to Stay'}")
    print(f"  │")
    print(f"  │  Key Factors:")
    if features_dict.get('GPA', 3.0) < 2.0:
        print(f"  │    ❌ Low GPA ({features_dict.get('GPA', 0):.2f} < 2.0)")
    if features_dict.get('Attendance_Rate', 100) < 75:
        print(f"  │    ❌ Low Attendance ({features_dict.get('Attendance_Rate', 0):.1f}% < 75%)")
    if features_dict.get('Stress_Index', 0) > 7:
        print(f"  │    ❌ High Stress (Level {features_dict.get('Stress_Index', 0):.1f}/10)")
    if features_dict.get('Part_Time_Job_Encoded', 0) == 1 and features_dict.get('Travel_Time_Minutes', 0) > 45:
        print(f"  │    ❌ Part-time job + Long commute")
    if features_dict.get('Scholarship_Encoded', 1) == 0:
        print(f"  │    ⚠️ No scholarship")
    
    print(f"  │")
    print(f"  │  Recommendations:")
    if prob > 0.5:
        print(f"  │    • Immediate academic counseling")
        print(f"  │    • Financial aid review")
        print(f"  │    • Study skills workshop")
        print(f"  │    • Stress management support")
    else:
        print(f"  │    • Continue current trajectory")
        print(f"  │    • Monitor attendance regularly")
    print(f"  └──────────────────────────────────────────────────\n")

def interactive_mode(model, scaler, meta):
    print("\n" + "="*55)
    print("  🎓 Student Dropout Risk Predictor — CLI")
    print(f"  Model: {meta['best_model']}  |  ROC-AUC: 0.812")
    print("="*55)
    print("  Type 'quit' to exit\n")
    
    while True:
        print("-"*45)
        try:
            age = input("  Age (17-30): ").strip()
            if age.lower() in ("quit","exit","q"): print("  Goodbye! 👋"); break
            age = float(age)
            
            gender = input("  Gender (Male/Female): ").strip()
            gender_enc = 1 if gender.lower() == "male" else 0
            
            income = float(input("  Family Income (NPR): ").strip() or "30000")
            internet = input("  Internet Access (Yes/No): ").strip().lower()
            internet_enc = 1 if internet == "yes" else 0
            
            study_hrs = float(input("  Study Hours/Day (1-10): ").strip() or "4")
            attendance = float(input("  Attendance Rate % (0-100): ").strip() or "80")
            delay_days = float(input("  Assignment Delay Days (0-10): ").strip() or "1")
            travel_time = float(input("  Travel Time Minutes (5-120): ").strip() or "30")
            
            job = input("  Part-time Job? (Yes/No): ").strip().lower()
            job_enc = 1 if job == "yes" else 0
            
            scholarship = input("  Has Scholarship? (Yes/No): ").strip().lower()
            scholarship_enc = 1 if scholarship == "yes" else 0
            
            stress = float(input("  Stress Index (1-10): ").strip() or "5")
            gpa = float(input("  GPA (0.0-4.0): ").strip() or "2.5")
            sem_gpa = float(input("  Semester GPA (0.0-4.0): ").strip() or str(gpa))
            cgpa = float(input("  CGPA (0.0-4.0): ").strip() or str(gpa))
            
            semester = input("  Semester (Year 1/2/3/4): ").strip()
            sem_enc = {"year 1":0, "year 2":1, "year 3":2, "year 4":3}.get(semester.lower(), 0)
            
            dept = input("  Department (Arts/Engineering/CS/Business/Science): ").strip()
            dept_enc = {"arts":0, "engineering":1, "cs":2, "business":3, "science":4}.get(dept.lower(), 2)
            
            edu = input("  Parental Education (None/High School/Bachelor/Master): ").strip()
            edu_enc = {"none":0, "high school":1, "bachelor":2, "master":3}.get(edu.lower(), 1)
            
            # Engineered features
            gpa_att = gpa * attendance
            stress_job = stress * job_enc
            income_sch = income * scholarship_enc
            study_att = study_hrs * attendance
            high_stress = 1 if stress > 7 else 0
            low_att = 1 if attendance < 75 else 0
            low_gpa = 1 if gpa < 2.0 else 0
            long_commute = 1 if travel_time > 45 else 0
            
            features = [age, gender_enc, income, internet_enc, study_hrs, attendance,
                       delay_days, travel_time, job_enc, scholarship_enc, stress, gpa,
                       sem_gpa, cgpa, sem_enc, dept_enc, edu_enc, gpa_att, stress_job,
                       income_sch, study_att, high_stress, low_att, low_gpa, long_commute]
            
            pred, prob = predict(features, model, scaler)
            
            features_dict = {
                'GPA': gpa, 'Attendance_Rate': attendance, 'Stress_Index': stress,
                'Part_Time_Job_Encoded': job_enc, 'Travel_Time_Minutes': travel_time,
                'Scholarship_Encoded': scholarship_enc
            }
            
            print_result(pred, prob, features_dict)
            
        except Exception as e:
            print(f"  ❌ Error: {e}. Please try again.")
            continue

def main():
    parser = argparse.ArgumentParser(description="Student Dropout Risk Predictor CLI")
    parser.add_argument("--gpa", type=float, default=None)
    parser.add_argument("--attendance", type=float, default=None)
    parser.add_argument("--stress", type=float, default=None)
    args = parser.parse_args()
    
    model, scaler, meta = load_model()
    
    if args.gpa and args.attendance:
        # Quick batch prediction
        features = [20, 1, 30000, 1, 4, args.attendance, 1, 30, 0, 0, args.stress or 5,
                   args.gpa, args.gpa, args.gpa, 0, 2, 1,
                   args.gpa*args.attendance, 0, 0, 4*args.attendance,
                   1 if args.stress and args.stress > 7 else 0,
                   1 if args.attendance < 75 else 0,
                   1 if args.gpa < 2.0 else 0, 0]
        pred, prob = predict(features, model, scaler)
        print_result(pred, prob, {'GPA': args.gpa, 'Attendance_Rate': args.attendance,
                                  'Stress_Index': args.stress or 5})
    else:
        interactive_mode(model, scaler, meta)

if __name__ == "__main__":
    main()
