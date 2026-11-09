# Spam Message Detection (NLP)

Classifies short text messages as **spam** or **ham** (legitimate) using classical NLP and machine learning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ujwalreddy18/Twitter-spam-detection/blob/main/spam_detection.ipynb)

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (`spam.csv`): 5,572 messages, of which 5,169 remain after removing 403 duplicates. **12.6% are spam**, so the classes are imbalanced.

## Approach

1. **Cleaning:** fix the file encoding and drop duplicate messages *before* splitting, so identical texts don't leak between train and test.
2. **Normalisation:** lowercase the text and replace URLs, e-mail addresses, money amounts and phone numbers/short codes with placeholder tokens (`urltoken`, `moneytoken`, `phonetoken`, …). Keep `!` and `?`.
3. **Features:** TF-IDF on word 1–2-grams combined with character 3–5-grams. The character n-grams catch obfuscated spellings such as "fr33" and "txt".
4. **Models:** Multinomial NB, Complement NB, Logistic Regression and Linear SVM. The linear models are class-weighted to handle the imbalance. A majority-class baseline is included for reference.
5. **Evaluation:** a stratified 80/20 split, 5-fold stratified cross-validation on the training set, then scoring on the held-out test set. The metrics are spam precision, recall and F1, ROC-AUC and PR-AUC.
6. **Error analysis and interpretation:** misclassified messages are inspected, and the top spam/ham features are read from the Logistic Regression coefficients.

## Results (held-out test set, 1,034 messages)

| Model | Accuracy | Spam precision | Spam recall | Spam F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Linear SVM** | **0.989** | **0.962** | **0.954** | **0.958** | **0.996** |
| Logistic Regression | 0.988 | 0.961 | 0.947 | 0.954 | 0.995 |
| Multinomial NB | 0.985 | 0.932 | 0.947 | 0.939 | 0.996 |
| Complement NB | 0.980 | 0.887 | 0.962 | 0.923 | 0.994 |
| Baseline (always "ham") | 0.874 | 0.000 | 0.000 | 0.000 | — |

The best model, Linear SVM, made 5 false positives and 6 false negatives out of 1,034 test messages. The baseline row shows why accuracy alone is misleading here: it scores 87% while catching no spam at all.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook spam_detection.ipynb
```

The notebook saves the final calibrated model to `spam_model.joblib`. Its `predict_message()` function returns a label and a spam probability:

```python
predict_message("URGENT! Your mobile number has won a free iPhone. Text CLAIM to 87121")
# ('SPAM', 1.0)
```

To use the saved model outside the notebook, copy the `normalise()` function from the notebook and apply it to the text before calling `model.predict_proba`.

## Limitations

- The dataset is a UK SMS corpus collected around 2012. Its slang and scam formats differ from modern messages and social-media posts.
- Detecting spam on Twitter/X would need tweet text plus account-level features such as followers, account age and posting rate.

## Next steps

- Fine-tune a small transformer (e.g. DistilBERT) and compare it against the linear baseline.
- Tune the decision threshold to hit a target false-positive rate.
- Serve the model behind a small API or Streamlit demo.
