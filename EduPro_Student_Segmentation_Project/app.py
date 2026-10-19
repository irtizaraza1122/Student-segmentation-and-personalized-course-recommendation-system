import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from pathlib import Path

st.set_page_config(page_title="EduPro Learner Intelligence", layout="wide")
st.title("EduPro | Student Segmentation & Course Recommendations")
st.caption("Behavior-based learner insights. Validate data and model results before operational use.")

DATA_DIR = Path("data")

@st.cache_data
def load_data():
    users = pd.read_csv(DATA_DIR / "users.csv")
    courses = pd.read_csv(DATA_DIR / "courses.csv")
    tx = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["TransactionDate"], dayfirst=False)
    required = {
        "users": {"UserID", "Age", "Gender"},
        "courses": {"CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating"},
        "transactions": {"UserID", "CourseID", "TransactionDate", "Amount"},
    }
    for name, df in [("users", users), ("courses", courses), ("transactions", tx)]:
        missing = required[name] - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv missing required columns: {sorted(missing)}")
    return users, courses, tx

def prepare(users, courses, tx):
    users = users.drop_duplicates("UserID").copy()
    courses = courses.drop_duplicates("CourseID").copy()
    tx = tx.copy()
    tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], errors="coerce")
    tx["Amount"] = pd.to_numeric(tx["Amount"], errors="coerce")
    tx = tx.dropna(subset=["UserID", "CourseID"])
    joined = tx.merge(users, on="UserID", how="left", indicator="user_join")
    joined = joined.merge(courses, on="CourseID", how="left", indicator="course_join")
    quality = {
        "transactions": len(tx),
        "unmatched_user_rows": int((joined["user_join"] == "left_only").sum()),
        "unmatched_course_rows": int((joined["course_join"] == "left_only").sum()),
        "missing_dates": int(joined["TransactionDate"].isna().sum()),
        "missing_amounts": int(joined["Amount"].isna().sum()),
    }
    joined = joined[(joined["user_join"] == "both") & (joined["course_join"] == "both")].copy()
    joined["CourseCategory"] = joined["CourseCategory"].fillna("Unknown").astype(str).str.strip()
    joined["CourseLevel"] = joined["CourseLevel"].fillna("Unknown").astype(str).str.strip()
    joined["CourseType"] = joined["CourseType"].fillna("Unknown").astype(str).str.strip()

    # One row per learner; use counts from observed transaction history.
    base = joined.groupby("UserID").agg(
        total_courses=("CourseID", "nunique"),
        total_transactions=("CourseID", "size"),
        total_spend=("Amount", "sum"),
        avg_spend=("Amount", "mean"),
        avg_course_rating=("CourseRating", "mean"),
        recency_date=("TransactionDate", "max"),
        category_diversity=("CourseCategory", "nunique"),
    )
    cat_counts = joined.groupby(["UserID", "CourseCategory"]).size().rename("n").reset_index()
    preferred_cat = cat_counts.sort_values(["UserID", "n", "CourseCategory"], ascending=[True, False, True]).drop_duplicates("UserID").set_index("UserID")["CourseCategory"]
    base["preferred_category"] = preferred_cat
    level_counts = joined.groupby(["UserID", "CourseLevel"]).size().rename("n").reset_index()
    preferred_level = level_counts.sort_values(["UserID", "n", "CourseLevel"], ascending=[True, False, True]).drop_duplicates("UserID").set_index("UserID")["CourseLevel"]
    base["preferred_level"] = preferred_level
    category_shares = cat_counts.assign(p=lambda d: d["n"] / d.groupby("UserID")["n"].transform("sum"))
    entropy = category_shares.assign(term=lambda d: -d["p"] * np.log(d["p"].clip(lower=1e-12))).groupby("UserID")["term"].sum()
    base["category_entropy"] = entropy
    latest = joined["TransactionDate"].max()
    base["recency_days"] = (latest - base["recency_date"]).dt.days if pd.notna(latest) else np.nan
    base = base.drop(columns=["recency_date"]).reset_index()
    profiles = users.merge(base, on="UserID", how="right")

    history = joined[["UserID", "CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating", "TransactionDate"]].copy()
    return profiles, history, joined, courses, quality

def fit_clusters(profiles, k):
    numeric = ["total_courses", "total_transactions", "total_spend", "avg_spend",
               "avg_course_rating", "category_diversity", "category_entropy", "recency_days"]
    categorical = ["preferred_category", "preferred_level"]
    numeric = [c for c in numeric if c in profiles.columns]
    categorical = [c for c in categorical if c in profiles.columns]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                               ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    preprocessor = ColumnTransformer(transformers)
    X = preprocessor.fit_transform(profiles)
    if X.shape[0] <= k:
        raise ValueError("Number of learners must exceed selected k.")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels, sample_size=min(3000, X.shape[0]), random_state=42) if len(set(labels)) > 1 and len(set(labels)) < len(labels) else np.nan
    profiles = profiles.copy()
    profiles["Segment"] = labels
    return profiles, model, preprocessor, X, score

def recommend(user_id, profiles, history, courses, top_n=10, category_filter=None, level_filter=None):
    row = profiles.loc[profiles["UserID"] == user_id]
    if row.empty:
        return pd.DataFrame()
    segment = row.iloc[0]["Segment"]
    seen = set(history.loc[history["UserID"] == user_id, "CourseID"])
    candidates = courses.loc[~courses["CourseID"].isin(seen)].copy()
    if category_filter and category_filter != "All":
        candidates = candidates[candidates["CourseCategory"].astype(str) == category_filter]
    if level_filter and level_filter != "All":
        candidates = candidates[candidates["CourseLevel"].astype(str) == level_filter]
    members = set(profiles.loc[profiles["Segment"] == segment, "UserID"])
    seg_history = history[history["UserID"].isin(members)]
    pop = seg_history.groupby("CourseID")["UserID"].nunique()
    cat = row.iloc[0].get("preferred_category", "Unknown")
    lvl = row.iloc[0].get("preferred_level", "Unknown")
    candidates["segment_popularity"] = candidates["CourseID"].map(pop).fillna(0)
    candidates["category_match"] = (candidates["CourseCategory"].astype(str) == str(cat)).astype(int)
    candidates["level_match"] = (candidates["CourseLevel"].astype(str) == str(lvl)).astype(int)
    rating = pd.to_numeric(candidates["CourseRating"], errors="coerce")
    rating_norm = (rating - rating.min()) / (rating.max() - rating.min()) if rating.max() != rating.min() else rating.fillna(0)
    pop_norm = candidates["segment_popularity"] / max(candidates["segment_popularity"].max(), 1)
    candidates["recommendation_score"] = (
        0.40 * candidates["category_match"] +
        0.20 * candidates["level_match"] +
        0.25 * pop_norm +
        0.15 * rating_norm.fillna(0)
    )
    candidates["reason"] = np.where(candidates["category_match"].eq(1), "Matches preferred category",
                            np.where(candidates["level_match"].eq(1), "Matches preferred level",
                                     "Popular or well-rated course"))
    return candidates.sort_values(["recommendation_score", "CourseRating"], ascending=False).head(top_n)

try:
    users, courses, tx = load_data()
    profiles, history, joined, courses, quality = prepare(users, courses, tx)
except Exception as e:
    st.error(f"Could not load project data: {e}")
    st.info("Place users.csv, courses.csv, and transactions.csv in the data/ directory. Column names must match the project brief.")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    k = st.slider("Number of clusters (k)", min_value=2, max_value=min(10, max(2, len(profiles)-1)), value=min(4, max(2, len(profiles)-1)))
    top_n = st.slider("Recommendations to show", 3, 20, 10)

try:
    clustered, model, preprocessor, X, sil = fit_clusters(profiles, k)
except Exception as e:
    st.error(f"Clustering could not run: {e}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Learners", f"{len(clustered):,}")
c2.metric("Courses", f"{courses['CourseID'].nunique():,}")
c3.metric("Transactions", f"{len(tx):,}")
c4.metric("Silhouette score", "N/A" if pd.isna(sil) else f"{sil:.3f}")
st.caption("Silhouette is an internal clustering metric, not a measure of recommendation quality or business impact.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Learner explorer", "Recommendations", "Data quality"])
with tab1:
    st.subheader("Segment sizes")
    sizes = clustered["Segment"].value_counts().sort_index()
    st.bar_chart(sizes)
    st.subheader("Segment profile (means)")
    numeric_cols = [c for c in ["total_courses","total_transactions","total_spend","avg_spend","avg_course_rating","category_diversity","category_entropy","recency_days"] if c in clustered]
    st.dataframe(clustered.groupby("Segment")[numeric_cols].mean().round(2), use_container_width=True)
    st.subheader("Candidate-k silhouette check")
    ks, scores = [], []
    for candidate_k in range(2, min(10, len(clustered)-1)+1):
        try:
            _, _, _, _, score = fit_clusters(profiles, candidate_k)
            ks.append(candidate_k); scores.append(score)
        except Exception:
            pass
    if scores:
        st.line_chart(pd.DataFrame({"k": ks, "silhouette": scores}).set_index("k"))
with tab2:
    ids = clustered["UserID"].astype(str).tolist()
    selected = st.selectbox("Select learner", ids)
    learner = clustered[clustered["UserID"].astype(str) == selected].iloc[0]
    st.write(f"**Assigned segment:** {int(learner['Segment'])}")
    show_cols = [c for c in ["Age","Gender","total_courses","total_transactions","total_spend","avg_spend","avg_course_rating","category_diversity","category_entropy","preferred_category","preferred_level","recency_days"] if c in learner.index]
    st.dataframe(pd.DataFrame({"Feature": show_cols, "Value": [learner[c] for c in show_cols]}), use_container_width=True)
    st.subheader("Enrollment history")
    st.dataframe(history[history["UserID"].astype(str) == selected].sort_values("TransactionDate", ascending=False), use_container_width=True)
with tab3:
    ids = clustered["UserID"].astype(str).tolist()
    selected = st.selectbox("Learner for recommendations", ids, key="rec_user")
    cats = ["All"] + sorted(courses["CourseCategory"].dropna().astype(str).unique().tolist())
    levels = ["All"] + sorted(courses["CourseLevel"].dropna().astype(str).unique().tolist())
    f1, f2 = st.columns(2)
    with f1: cat_filter = st.selectbox("Filter by category", cats)
    with f2: level_filter = st.selectbox("Filter by level", levels)
    recs = recommend(clustered.loc[clustered["UserID"].astype(str)==selected, "UserID"].iloc[0],
                     clustered, history, courses, top_n, cat_filter, level_filter)
    if recs.empty:
        st.info("No unseen courses match the current filters.")
    else:
        cols = [c for c in ["CourseID","CourseCategory","CourseType","CourseLevel","CourseRating","recommendation_score","reason"] if c in recs]
        st.dataframe(recs[cols].round(3), use_container_width=True)
with tab4:
    st.subheader("Join and data-quality checks")
    st.json(quality)
    st.write("Missing values by source:")
    st.dataframe(pd.DataFrame({
        "users_missing": users.isna().sum(),
        "courses_missing": courses.isna().sum(),
        "transactions_missing": tx.isna().sum()
    }).fillna(0).astype(int), use_container_width=True)
    st.warning("Review unmatched joins, duplicate keys, invalid amounts/dates, and course-rating scale before interpreting results.")
