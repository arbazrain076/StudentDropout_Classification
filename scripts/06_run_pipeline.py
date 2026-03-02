"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 06: Full Pipeline Runner
=============================================================================
Run : python scripts/06_run_pipeline.py
=============================================================================
"""

import subprocess, sys, time

STEPS = [
    ("scripts/01_preprocess.py",    "Data Preprocessing"),
    ("scripts/02_eda_minimal.py",   "Exploratory Data Analysis"),
    ("scripts/03_train_evaluate.py","Model Training & Evaluation"),
]

print("=" * 65)
print("  STW5000CEM — Student Dropout Prediction")
print("  Full Pipeline Runner")
print("=" * 65)

total_start = time.time()
for script, label in STEPS:
    print(f"\n{'─'*65}\n  ▶  {label}\n{'─'*65}")
    t = time.time()
    r = subprocess.run([sys.executable, script], capture_output=False)
    if r.returncode != 0:
        print(f"\n  ❌ FAILED: {label}")
        sys.exit(1)
    print(f"\n  ✅ Done  ({time.time()-t:.1f}s)")

print(f"\n{'='*65}")
print(f"  ✅ Full pipeline complete in {time.time()-total_start:.1f}s")
print(f"{'='*65}")
print("\n  To predict dropout risk:")
print("    python scripts/05_cli.py")
print("    python scripts/05_cli.py --gpa 1.8 --attendance 65 --stress 8")
