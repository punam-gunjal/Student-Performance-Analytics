#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STUDENT PERFORMANCE ANALYTICS SYSTEM
# Technologies: Python, NumPy, Pandas, Matplotlib
# ============================================================

# 1. LOAD DATASET
DATA_FILE = Path(r"C:\Users\om\Downloads\students.csv")

SUBJECTS = ["Python", "AI", "Statistics", "Data_Analysis"]
ATTENDANCE_COLUMN = "Attendance_Percentage"

PASS_MARKS = 40
MIN_ATTENDANCE = 75

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Check the CSV file location and name."
    )

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("=" * 65)
print("          STUDENT PERFORMANCE ANALYTICS SYSTEM")
print("=" * 65)

# 2. DISPLAY DATASET INFORMATION
print("\n1. DATASET INFORMATION")
print("-" * 65)

print(f"Total students: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("\nAvailable columns:")
print(df.columns.tolist())

# 3. VALIDATE REQUIRED COLUMNS
required_columns = [
    "Student_ID",
    "Name",
    "Department",
    *SUBJECTS,
    ATTENDANCE_COLUMN
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

if df.empty:
    raise ValueError("The dataset contains no student records.")

# 4. VALIDATE NUMERIC VALUES
numeric_columns = SUBJECTS + [ATTENDANCE_COLUMN]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

if df[numeric_columns].isnull().any().any():
    raise ValueError(
        "Missing or invalid numeric values found in subject marks "
        "or attendance. Please check the CSV dataset."
    )

if not df[SUBJECTS].apply(
    lambda column: column.between(0, 100).all()
).all():
    raise ValueError("Subject marks must be between 0 and 100.")

if not df[ATTENDANCE_COLUMN].between(0, 100).all():
    raise ValueError("Attendance must be between 0 and 100.")

# 5. CALCULATE TOTAL AND AVERAGE MARKS
df["Total_Marks"] = df[SUBJECTS].sum(axis=1)

df["Average_Marks"] = df[SUBJECTS].mean(axis=1).round(2)

# 6. ASSIGN GRADES
def calculate_grade(average):
    if average >= 90:
        return "A+"
    elif average >= 80:
        return "A"
    elif average >= 70:
        return "B"
    elif average >= 60:
        return "C"
    elif average >= 50:
        return "D"
    elif average >= 40:
        return "E"
    else:
        return "F"

df["Grade"] = df["Average_Marks"].apply(calculate_grade)

# 7. DETERMINE PASS OR FAIL
# Passing requires at least 40 marks in every subject
# and at least 75% attendance.

df["Result"] = np.where(
    (df[SUBJECTS].ge(PASS_MARKS).all(axis=1))
    & (df[ATTENDANCE_COLUMN] >= MIN_ATTENDANCE),
    "Pass",
    "Fail"
)

# 8. CALCULATE STUDENT RANKS
df["Rank"] = (
    df["Total_Marks"]
    .rank(method="min", ascending=False)
    .astype(int)
)

# 9. DISPLAY STUDENT-WISE PERFORMANCE
print("\n\n2. STUDENT-WISE PERFORMANCE")
print("-" * 65)

student_columns = [
    "Student_ID",
    "Name",
    *SUBJECTS,
    ATTENDANCE_COLUMN,
    "Total_Marks",
    "Average_Marks",
    "Grade",
    "Result",
    "Rank"
]

print(
    df[student_columns]
    .sort_values(["Rank", "Name"])
    .to_string(index=False)
)

# 10. CLASS PERFORMANCE SUMMARY
class_average = df["Average_Marks"].mean()

highest_average = df["Average_Marks"].max()
lowest_average = df["Average_Marks"].min()

highest_student = df.loc[df["Average_Marks"].idxmax()]
lowest_student = df.loc[df["Average_Marks"].idxmin()]

pass_count = int((df["Result"] == "Pass").sum())
fail_count = int((df["Result"] == "Fail").sum())

print("\n\n3. CLASS PERFORMANCE SUMMARY")
print("-" * 65)

print(f"Total students       : {len(df)}")
print(f"Class average        : {class_average:.2f}")
print(f"Highest average      : {highest_average:.2f}")
print(f"Highest scorer       : {highest_student['Name']}")
print(f"Lowest average       : {lowest_average:.2f}")
print(f"Lowest scorer        : {lowest_student['Name']}")
print(f"Students passed      : {pass_count}")
print(f"Students failed      : {fail_count}")
print(
    f"Pass percentage      : "
    f"{(pass_count / len(df)) * 100:.2f}%"
)

# 11. SUBJECT-WISE ANALYSIS
subject_summary = pd.DataFrame({
    "Subject": SUBJECTS,
    "Average_Marks": df[SUBJECTS].mean().round(2).values,
    "Highest_Marks": df[SUBJECTS].max().values,
    "Lowest_Marks": df[SUBJECTS].min().values,
    "Pass_Count": df[SUBJECTS].ge(PASS_MARKS).sum().values
})

subject_summary["Pass_Percentage"] = (
    subject_summary["Pass_Count"] / len(df) * 100
).round(2)

print("\n\n4. SUBJECT-WISE ANALYSIS")
print("-" * 65)

print(subject_summary.to_string(index=False))

best_subject = subject_summary.loc[
    subject_summary["Average_Marks"].idxmax(),
    "Subject"
]

weakest_subject = subject_summary.loc[
    subject_summary["Average_Marks"].idxmin(),
    "Subject"
]

print(f"\nBest-performing subject : {best_subject}")
print(f"Lowest-average subject  : {weakest_subject}")

# 12. DISPLAY TOP FIVE PERFORMERS
print("\n\n5. TOP FIVE PERFORMERS")
print("-" * 65)

top_five = df.sort_values(
    ["Total_Marks", "Name"],
    ascending=[False, True]
).head(5)

print(
    top_five[
        [
            "Rank",
            "Student_ID",
            "Name",
            "Total_Marks",
            "Average_Marks",
            "Grade",
            "Result"
        ]
    ].to_string(index=False)
)

# 13. DISPLAY PASS STUDENTS
print("\n\n6. PASS STUDENTS")
print("-" * 65)

passed_students = df[df["Result"] == "Pass"]

if passed_students.empty:
    print("No students met both passing criteria.")
else:
    print(
        passed_students[
            [
                "Student_ID",
                "Name",
                "Average_Marks",
                ATTENDANCE_COLUMN,
                "Grade"
            ]
        ].to_string(index=False)
    )

# 14. DISPLAY FAIL STUDENTS
print("\n\n7. FAIL STUDENTS")
print("-" * 65)

failed_students = df[df["Result"] == "Fail"]

if failed_students.empty:
    print("All students passed.")
else:
    print(
        failed_students[
            [
                "Student_ID",
                "Name",
                "Average_Marks",
                ATTENDANCE_COLUMN,
                "Grade"
            ]
        ].to_string(index=False)
    )

# 15. GRADE DISTRIBUTION
grade_order = ["A+", "A", "B", "C", "D", "E", "F"]

grade_distribution = (
    df["Grade"]
    .value_counts()
    .reindex(grade_order, fill_value=0)
)

print("\n\n8. GRADE DISTRIBUTION")
print("-" * 65)

print(grade_distribution.to_string())

# 16. GRAPH 1: SUBJECT-WISE AVERAGE MARKS
plt.figure(figsize=(9, 5))

plt.bar(
    subject_summary["Subject"],
    subject_summary["Average_Marks"]
)

plt.title("Subject-Wise Average Marks")
plt.xlabel("Subjects")
plt.ylabel("Average Marks")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# 17. GRAPH 2: GRADE DISTRIBUTION
plt.figure(figsize=(8, 5))

plt.bar(
    grade_distribution.index,
    grade_distribution.values
)

plt.title("Student Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# 18. COMPLETION MESSAGE
print("\n" + "=" * 65)
print("       STUDENT PERFORMANCE ANALYSIS COMPLETED")
print("=" * 65)


# In[ ]:




