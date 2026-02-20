import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_spam_dataset(file_path: str) -> pd.DataFrame:
    """
    Handles your spam.csv where header is 'v1,v2' (comma) but rows are TAB-separated.
    Also supports rows that might be comma-separated.
    """
    rows = []
    with open(file_path, "r", encoding="latin-1", errors="replace") as f:
        header = f.readline()  # skip header (v1,v2)

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Most rows: label \t text
            if "\t" in line:
                label, text = line.split("\t", 1)

            # Fallback: label,text
            elif "," in line and (line.lower().startswith("ham,") or line.lower().startswith("spam,")):
                label, text = line.split(",", 1)
            else:
                continue

            label = label.strip().lower()
            text = text.strip()

            if label in ("ham", "spam") and text:
                rows.append((label, text))

    df = pd.DataFrame(rows, columns=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def main():
    os.makedirs("model", exist_ok=True)

    data_path = os.path.join("data", "spam.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_spam_dataset(data_path)
    if len(df) < 50:
        raise ValueError("Dataset still looks too small. Your spam.csv may be different/corrupted.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Training complete | Accuracy: {acc*100:.2f}%")
    print(f"✅ Rows used: {len(df)}")

    joblib.dump(vectorizer, os.path.join("model", "vectorizer.pkl"))
    joblib.dump(model, os.path.join("model", "model.pkl"))

    print("✅ Saved:")
    print("   - model/vectorizer.pkl")
    print("   - model/model.pkl")


if __name__ == "__main__":
    main()
