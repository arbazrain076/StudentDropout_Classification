"""EDA + Charts Generation for Student Dropout"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
plt.style.use('seaborn-v0_8-whitegrid'); os.makedirs('charts', exist_ok=True)
df = pd.read_csv('data/clean/student_dropout_clean.csv')

# Class distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
counts = df['Dropout'].value_counts()
ax[0].bar(['Retained', 'Dropped'], counts.values, color=['#27ae60', '#e74c3c'])
ax[0].set_title('Class Distribution'); ax[0].set_ylabel('Count')
ax[1].pie(counts.values, labels=['Retained', 'Dropped'], autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'])
plt.tight_layout(); plt.savefig('charts/class_distribution.png', dpi=150); plt.close()
print('✓ Saved: class_distribution.png')

# GPA vs Attendance
fig, ax = plt.subplots(figsize=(10, 6))
for dropout, color in [(0, '#27ae60'), (1, '#e74c3c')]:
    sub = df[df['Dropout'] == dropout]
    ax.scatter(sub['GPA'], sub['Attendance_Rate'], alpha=0.5, c=color, 
               label='Dropped' if dropout else 'Retained', s=30)
ax.set_xlabel('GPA'); ax.set_ylabel('Attendance Rate (%)')
ax.set_title('GPA vs Attendance by Dropout Status'); ax.legend()
plt.tight_layout(); plt.savefig('charts/gpa_vs_attendance.png', dpi=150); plt.close()
print('✓ Saved: gpa_vs_attendance.png')

print('EDA complete ✅')
