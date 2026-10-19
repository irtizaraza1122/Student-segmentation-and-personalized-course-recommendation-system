# EduPro Student Segmentation & Recommendation System

## 1. Prepare the data
Export the three workbook tabs to CSV and save as:
- `data/users.csv`
- `data/courses.csv`
- `data/transactions.csv`

Required columns:
- Users: `UserID, Age, Gender`
- Courses: `CourseID, CourseCategory, CourseType, CourseLevel, CourseRating`
- Transactions: `UserID, CourseID, TransactionDate, Amount`

Check column names and date formats before running. The application parses `TransactionDate` with pandas and reports unmatched joins.

## 2. Install and run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 3. Important caveats
- The report is a methodology and implementation blueprint until run on the actual workbook.
- Cluster IDs are arbitrary; inspect feature profiles before naming segments.
- The recommendation scoring weights are initial heuristics and must be tuned using temporal holdout evaluation.
- CourseRating is treated as a course-level signal, not an individual learner rating.
- Silhouette is not recommendation precision or business impact.
- Do not use demographic fields to restrict course access.
