"""Train the deployable model and export it as JSON for the website."""
import json, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


def normalise(text):
    t = text.lower()
    t = re.sub(r"(https?://\S+|www\.\S+|\S+\.(com|net|org|co\.uk)\S*)", " urltoken ", t)
    t = re.sub(r"\S+@\S+", " emailtoken ", t)
    t = re.sub(r"[£$€]\s?\d+(\.\d+)?|\d+(\.\d+)?\s?(p|pence|pounds?)\b", " moneytoken ", t)
    t = re.sub(r"\b\d{5,}\b", " phonetoken ", t)
    t = re.sub(r"\d+", " numtoken ", t)
    t = re.sub(r"[^a-z!? ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def make():
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=3000)),
    ])


df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
df["text"] = df["text"].str.replace("å£", "£", regex=False)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
df["y"] = (df.label == "spam").astype(int)
df["clean"] = df.text.apply(normalise)

# Held-out metrics for the site
Xtr, Xte, ytr, yte = train_test_split(df.clean, df.y, test_size=0.2, stratify=df.y, random_state=42)
m = make().fit(Xtr, ytr)
p = m.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
metrics = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
               recall=recall_score(yte, pred), f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, p),
               n_test=int(len(yte)), fp=int(((pred == 1) & (yte == 0)).sum()), fn=int(((pred == 0) & (yte == 1)).sum()))
print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

# Final model on all data
final = make().fit(df.clean, df.y)
union = final.named_steps["tfidf"]
coef = final.named_steps["clf"].coef_[0]
blocks, off = {}, 0
for name, vec in union.transformer_list:
    vocab = vec.vocabulary_
    n = len(vocab)
    terms = [None] * n
    for t, i in vocab.items():
        terms[i] = t
    blocks[name] = {"terms": terms,
                    "idf": [round(float(x), 5) for x in vec.idf_],
                    "coef": [round(float(x), 5) for x in coef[off:off + n]]}
    off += n
model = {"intercept": float(final.named_steps["clf"].intercept_[0]), **blocks, "metrics": metrics,
         "n_messages": int(len(df)), "n_spam": int(df.y.sum())}
json.dump(model, open("model.json", "w"), separators=(",", ":"))

# Reference probabilities for parity testing
ref = df[["text"]].copy(); ref["p"] = final.predict_proba(df.clean)[:, 1]
ref.to_json("ref.json", orient="records")
print("word feats", len(blocks["word"]["terms"]), "char feats", len(blocks["char"]["terms"]))
