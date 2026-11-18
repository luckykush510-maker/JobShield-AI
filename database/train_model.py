import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("fake_job_postings.csv")

print("Dataset loaded successfully.")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))


# ==========================================
# 2. CHECK TARGET COLUMN
# ==========================================

print("\nColumns:")
print(df.columns.tolist())

if "fraudulent" not in df.columns:
    print("\nERROR: 'fraudulent' column not found.")
    exit()


# ==========================================
# 3. SELECT USEFUL TEXT COLUMNS
# ==========================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location",
    "salary_range"
]


# Only use columns that actually exist
available_columns = [
    column for column in text_columns
    if column in df.columns
]


print("\nText columns being used:")
print(available_columns)


# ==========================================
# 4. FILL MISSING VALUES
# ==========================================

for column in available_columns:
    df[column] = df[column].fillna("")


# ==========================================
# 5. COMBINE TEXT
# ==========================================

df["combined_text"] = df[available_columns].astype(str).agg(
    " ".join,
    axis=1
)


# ==========================================
# 6. REMOVE EMPTY TEXT
# ==========================================

df = df[df["combined_text"].str.strip() != ""]


# ==========================================
# 7. INPUT AND TARGET
# ==========================================

X = df["combined_text"]
y = df["fraudulent"]


print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 8. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 9. TF-IDF
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF created successfully.")


# ==========================================
# 10. TRAIN ML MODEL
# ==========================================

print("\nTraining ML model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

print("Model training completed.")


# ==========================================
# 11. PREDICTION
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 12. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fake"],
    zero_division=0
))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 13. SAVE MODEL
# ==========================================

print("\nSaving model...")

joblib.dump(
    model,
    "model/fake_job_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


print("\n==========================================")
print("SUCCESS")
print("==========================================")

print("Model saved:")
print("model/fake_job_model.pkl")

print("Vectorizer saved:")
print("model/tfidf_vectorizer.pkl")

print("\nJobShield AI ML model is ready.")