# STEP 1: Import required packages
import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
from sentence_transformers import SentenceTransformer

# STEP 2: Load the labelled dataset
LABELED_DATASET = "patents_classified_500_gpt5_mini.xlsx"

df = pd.read_excel(LABELED_DATASET)

# STEP 3: Combine title and abstract into a single input text
df["text"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")


# STEP 4: Utility functions
def isco3_to_isco2(code):
    """Convert a 3-digit ISCO code to 2-digit"""
    if pd.isna(code):
        return None
    code_str = str(code).strip()
    code_str = re.sub(r"\D", "", code_str)
    return code_str[:2] if len(code_str) >= 2 else None


def parse_secondary_codes(val):
    """Parse stringified list to actual list of secondary codes"""
    try:
        codes = ast.literal_eval(val) if isinstance(val, str) else val
        return codes if isinstance(codes, list) else []
    except Exception:
        return []


# STEP 5: Build a list of ISCO-2 labels per patent (primary + secondary)
df["labels"] = df.apply(
    lambda row: sorted(
        set(
            [isco3_to_isco2(row["primary_code"])]
            + [isco3_to_isco2(c) for c in parse_secondary_codes(row["secondary_codes"])]
        )
    ),
    axis=1,
)

# STEP 6: Remove rows with no valid labels
df = df[df["labels"].apply(lambda x: len([l for l in x if l]) > 0)].reset_index(
    drop=True
)

# STEP 7: Multi-label binarization
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["labels"])
class_names = mlb.classes_

# STEP 8: Text embedding using sentence-transformers
model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model
X = model.encode(df["text"].tolist(), show_progress_bar=True)

# STEP 9: Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# STEP 10: One-vs-Rest Logistic Regression classifier
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
clf.fit(X_train, Y_train)

# STEP 11: Predictions
Y_pred = clf.predict(X_test)

# STEP 12: Evaluation
micro_f1 = f1_score(Y_test, Y_pred, average="micro")
macro_f1 = f1_score(Y_test, Y_pred, average="macro")
report = classification_report(Y_test, Y_pred, target_names=class_names)

# Print results
print("✅ Micro-F1 score:", micro_f1)
print("✅ Macro-F1 score:", macro_f1)
print("\n📋 Classification Report:\n", report)
