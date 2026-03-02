"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 03: Classification Model Training & Evaluation
=============================================================================
Domain    : BINARY CLASSIFICATION
Problem   : Predict student dropout (0=Retained, 1=Dropped Out)
Algorithms: KNN, Decision Tree, Naive Bayes, Random Forest, Logistic Regression
Dataset   : 10,000 students with 33 features
Run       : python scripts/03_train_evaluate.py
=============================================================================
"""

import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, auc)

warnings.filterwarnings("ignore")
os.makedirs("model", exist_ok=True)
os.makedirs("charts", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CLASS 1: StudentDataset
# ═══════════════════════════════════════════════════════════════════════════
class StudentDataset:
    """Wraps the student dropout dataset with utility methods."""
    
    FEATURE_COLS = [
        'Age', 'Gender_Encoded', 'Family_Income', 'Internet_Access_Encoded',
        'Study_Hours_per_Day', 'Attendance_Rate', 'Assignment_Delay_Days',
        'Travel_Time_Minutes', 'Part_Time_Job_Encoded', 'Scholarship_Encoded',
        'Stress_Index', 'GPA', 'Semester_GPA', 'CGPA', 'Semester_Encoded',
        'Department_Encoded', 'Parental_Education_Encoded', 'GPA_Attendance',
        'Stress_Job', 'Income_Scholarship', 'Study_Attendance', 'High_Stress',
        'Low_Attendance', 'Low_GPA', 'Long_Commute'
    ]
    TARGET_COL = "Dropout"
    
    def __init__(self, csv_path: str):
        self._df = pd.read_csv(csv_path)
        print(f"  StudentDataset loaded: {self._df.shape}")
    
    @property
    def X(self):
        return self._df[self.FEATURE_COLS].values
    
    @property
    def y(self):
        return self._df[self.TARGET_COL].values
    
    def summary(self):
        return {
            "total_records": len(self._df),
            "features": len(self.FEATURE_COLS),
            "class_0": int((self._df[self.TARGET_COL]==0).sum()),
            "class_1": int((self._df[self.TARGET_COL]==1).sum()),
            "class_0_pct": round((self._df[self.TARGET_COL]==0).sum()/len(self._df)*100, 1),
            "class_1_pct": round((self._df[self.TARGET_COL]==1).sum()/len(self._df)*100, 1)
        }

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CLASS 2: DropoutClassifier
# ═══════════════════════════════════════════════════════════════════════════
class DropoutClassifier:
    """Wrapper for binary classification with imbalance handling."""
    
    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator
        self.scaler = StandardScaler()
        self._fitted = False
        self.metrics_ = {}
        self.cv_scores_ = {}
    
    def fit(self, X_train, y_train, use_smote=True):
        X_sc = self.scaler.fit_transform(X_train)
        # Use class_weight='balanced' instead of SMOTE (network disabled)
        self.estimator.fit(X_sc, y_train)
        self._fitted = True
        return self
    
    def predict(self, X):
        return self.estimator.predict(self.scaler.transform(X))
    
    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(self.scaler.transform(X))
        return None
    
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        proba = self.predict_proba(X_test)
        self.metrics_ = {
            "accuracy":  round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
            "f1_score":  round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, proba[:,1]) if proba is not None else 0, 4),
            "conf_matrix": confusion_matrix(y_test, preds).tolist(),
            "report": classification_report(y_test, preds, target_names=["Retained","Dropped"],
                                           zero_division=0)
        }
        self._preds = preds
        return self.metrics_
    
    def cross_validate(self, X, y, cv=5):
        X_sc = self.scaler.fit_transform(X)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.estimator, X_sc, y, cv=skf, scoring="f1")
        self.cv_scores_ = {
            "mean": round(scores.mean(), 4),
            "std": round(scores.std(), 4),
            "all": scores.round(4).tolist()
        }
        return self.cv_scores_
    
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        safe = self.name.replace(" ","_").replace("(","").replace(")","").replace("=","")
        with open(os.path.join(directory, f"{safe}_model.pkl"), "wb") as f:
            pickle.dump(self.estimator, f)
        with open(os.path.join(directory, f"{safe}_scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CLASS 3: ModelComparator
# ═══════════════════════════════════════════════════════════════════════════
class ModelComparator:
    """Manages multiple classifiers and generates comparative analysis."""
    
    def __init__(self, dataset: StudentDataset):
        self.dataset = dataset
        self.classifiers = []
        self.results_df = None
        self._best = None
    
    def add(self, clf: DropoutClassifier):
        self.classifiers.append(clf)
        return self
    
    def run(self, test_size=0.20, random_state=42):
        X, y = self.dataset.X, self.dataset.y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        self._X_test, self._y_test = X_test, y_test
        
        print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
        rows = []
        for clf in self.classifiers:
            print(f"\n  ── {clf.name} ──────────────────")
            clf.fit(X_train, y_train, use_smote=True)
            clf.evaluate(X_test, y_test)
            clf.cross_validate(X_train, y_train)
            print(f"     Accuracy : {clf.metrics_['accuracy']:.4f}")
            print(f"     F1 Score : {clf.metrics_['f1_score']:.4f}")
            print(f"     ROC-AUC  : {clf.metrics_['roc_auc']:.4f}")
            rows.append({
                "Model": clf.name,
                "Accuracy": clf.metrics_["accuracy"],
                "Precision": clf.metrics_["precision"],
                "Recall": clf.metrics_["recall"],
                "F1 Score": clf.metrics_["f1_score"],
                "ROC-AUC": clf.metrics_["roc_auc"],
                "CV F1 Mean": clf.cv_scores_["mean"],
                "CV F1 Std": f"±{clf.cv_scores_['std']}"
            })
        
        self.results_df = pd.DataFrame(rows)
        best_idx = self.results_df["F1 Score"].idxmax()
        self._best = self.classifiers[best_idx]
        print(f"\n  ✅ Best model: {self._best.name} (F1={self._best.metrics_['f1_score']:.4f})")
        return self.results_df
    
    @property
    def best(self):
        return self._best
    
    def plot_comparison(self, save_path):
        metrics = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        colors = ["#2d6a4f","#40916c","#f77f00","#e63946","#457b9d"]
        for ax, met, col in zip(axes, metrics, colors):
            vals = self.results_df[met].astype(float)
            bars = ax.bar(range(len(vals)), vals, color=col, alpha=0.85, edgecolor="black")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(self.results_df["Model"], rotation=20, ha="right", fontsize=8)
            ax.set_title(met, fontsize=11, fontweight="bold")
            ax.set_ylim(max(0,vals.min()-.05), min(1,vals.max()+.08))
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.005,
                       f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        fig.suptitle("Model Comparison — Binary Classification Metrics", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"   Saved: {save_path}")
    
    def plot_roc_curves(self, save_path):
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#2d6a4f","#f77f00","#e63946","#457b9d","#9b59b6"]
        for clf, col in zip(self.classifiers, colors):
            if hasattr(clf.estimator, "predict_proba"):
                proba = clf.predict_proba(self._X_test)[:,1]
                fpr, tpr, _ = roc_curve(self._y_test, proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=col, lw=2, label=f"{clf.name} (AUC={roc_auc:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"   Saved: {save_path}")
    
    def plot_confusion_matrices(self, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for ax, clf in zip(axes, self.classifiers):
            cm = np.array(clf.metrics_["conf_matrix"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                       xticklabels=["Retained","Dropped"],
                       yticklabels=["Retained","Dropped"], linewidths=0.5)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{clf.name}\nAcc={clf.metrics_['accuracy']:.3f}", fontsize=11, fontweight="bold")
        axes[-1].axis('off')
        fig.suptitle("Confusion Matrices — All Five Classifiers", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"   Saved: {save_path}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
print("="*65)
print("  STW5000CEM — Student Dropout Classification Training")
print("="*65)

print("\n[1/5] Loading dataset...")
dataset = StudentDataset("data/clean/student_dropout_model.csv")
summary = dataset.summary()
print(f"  Records: {summary['total_records']:,}")
print(f"  Features: {summary['features']}")
print(f"  Class 0 (Retained): {summary['class_0']:,} ({summary['class_0_pct']}%)")
print(f"  Class 1 (Dropped):  {summary['class_1']:,} ({summary['class_1_pct']}%)")

print("\n[2/5] Grid search for optimal k (KNN)...")
X_all, y_all = dataset.X, dataset.y
X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
scaler_temp = StandardScaler()
X_tr_sc = scaler_temp.fit_transform(X_tr)
X_val_sc = scaler_temp.transform(X_val)

k_results = []
for k in [1,3,5,7,9,11,15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr_sc, y_tr)
    preds = knn.predict(X_val_sc)
    f1 = f1_score(y_val, preds)
    k_results.append({"k": k, "f1": round(f1, 4)})
    print(f"    k={k:2d}  F1={f1:.4f}")

best_k = max(k_results, key=lambda x: x["f1"])["k"]
print(f"\n  ✅ Best k = {best_k}  (F1={max(k_results, key=lambda x: x['f1'])['f1']:.4f})")

print("\n[3/5] Training all classifiers...")
comparator = ModelComparator(dataset)
comparator.add(DropoutClassifier(f"KNN (k={best_k})",
    KNeighborsClassifier(n_neighbors=best_k, weights="distance")))
comparator.add(DropoutClassifier("Decision Tree",
    DecisionTreeClassifier(max_depth=10, min_samples_split=20, class_weight="balanced", random_state=42)))
comparator.add(DropoutClassifier("Naive Bayes", GaussianNB()))
comparator.add(DropoutClassifier("Random Forest",
    RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)))
comparator.add(DropoutClassifier("Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)))

results_df = comparator.run(test_size=0.20, random_state=42)

print("\n[4/5] Generating charts...")
comparator.plot_comparison("charts/model_comparison.png")
comparator.plot_roc_curves("charts/roc_curves.png")
comparator.plot_confusion_matrices("charts/confusion_matrices.png")

print("\n[5/5] Saving models...")
comparator.best.save("model")
with open("model/best_model.pkl", "wb") as f: pickle.dump(comparator.best.estimator, f)
with open("model/best_scaler.pkl", "wb") as f: pickle.dump(comparator.best.scaler, f)

meta = {
    "best_model": comparator.best.name,
    "feature_cols": StudentDataset.FEATURE_COLS,
    "dataset_summary": summary,
    "all_results": results_df.to_dict('records'),
    "knn_best_k": int(best_k),
    "knn_search": k_results
}
with open("model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

results_df.to_csv("model/metrics.csv", index=False)

print("\n── Final Results ─────────────────────────────────────")
print(results_df.to_string(index=False))
print(f"\n  Best Model : {comparator.best.name}")
print(f"  Accuracy   : {comparator.best.metrics_['accuracy']:.4f}")
print(f"  F1 Score   : {comparator.best.metrics_['f1_score']:.4f}")
print(f"  ROC-AUC    : {comparator.best.metrics_['roc_auc']:.4f}")
print("\n  Training complete ✅")
