from pathlib import Path
from importlib.metadata import version
import hashlib
import io
import re
import tarfile
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split

SEED = 2026
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
CLASS_NAMES = ['Negative', 'Positive']  # 0 = Negative, 1 = Positive
NEG, POS, INK = '#d26936', '#286fb5', '#233247'
CLASS_COLORS = [NEG, POS]
# plot style
plt.rcParams.update({
    'figure.figsize': (9, 4.3), 'figure.dpi': 115,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.titlesize': 12,
    'font.size': 10.5, 'axes.axisbelow': True,
    'grid.color': '#e4e7ec', 'grid.alpha': 0.8,
})
print('Package versions:')
for package in ['scikit-learn', 'numpy', 'scipy', 'pandas', 'matplotlib']:
    print(f'{package}: {version(package)}')


#________________________________________________________________________________________________

DATA_URL = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz'
ARCHIVE = DATA_DIR / 'review_polarity.tar.gz'
EXPECTED_SHA256 = 'fc0dccc2671af5db3c5d8f81f77a1ebfec953ecdd422334062df61ede36b2179'
# download the Cornell archive once and check it is the correct file
if not ARCHIVE.exists():
    try:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
    except OSError as exc:
        raise RuntimeError('Put the supplied review_polarity.tar.gz in data/ and rerun.') from exc
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Unexpected download checksum.')
    ARCHIVE.write_bytes(payload)
payload = ARCHIVE.read_bytes()
assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256
print('Verified original Cornell archive:', len(payload), 'bytes')

#________________________________________________________________________________________________

# read every review from the archive (label 1 = pos, 0 = neg)
rows = []
with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
    for member in sorted(archive.getmembers(), key=lambda item: item.name):
        parts = Path(member.name).parts
        if (member.isfile() and len(parts) == 3 and parts[0] == 'txt_sentoken'
                and parts[1] in {'neg', 'pos'} and parts[2].endswith('.txt')):
            rows.append({'text': archive.extractfile(member).read().decode('utf-8'),
                         'label': int(parts[1] == 'pos'), 'file': member.name})
reviews = pd.DataFrame(rows)
# basic checks: 2000 reviews, 1000 per class, no empty texts, no duplicates
assert reviews.shape[0] == 2000
assert reviews.label.value_counts().sort_index().tolist() == [1000, 1000]
assert reviews.text.str.strip().ne('').all()
dedup_text = reviews.text.str.replace(r'\s+', ' ', regex=True).str.strip()
assert not dedup_text.duplicated().any(), 'Resolve duplicated texts before splitting.'

# keep 30% aside as the test set, used only at the end
trainval, sealed_test = train_test_split(
    reviews, test_size=0.30, stratify=reviews.label, random_state=SEED)
trainval = trainval.reset_index(drop=True)
X_trainval = trainval.text.to_numpy()
y_trainval = trainval.label.to_numpy()
print(f'{len(reviews)} reviews: {len(trainval)} training + validation; {len(sealed_test)} sealed test.')
print('Class mapping:', dict(enumerate(CLASS_NAMES)))
print('Training + validation review excerpt:', X_trainval[0][:280].replace('\n', ' '))

#________________________________________________________________________________________________
# plot class balance and review lengths
review_lengths = trainval.text.str.split().str.len().to_numpy()
counts = np.bincount(y_trainval)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={'width_ratios': [1, 2]})
bars = axes[0].bar(CLASS_NAMES, counts, color=CLASS_COLORS, width=0.6)
axes[0].bar_label(bars, padding=3)
axes[0].set(title='Training + validation class balance', ylabel='Reviews', ylim=(0, counts.max() * 1.15))
bins = np.linspace(0, review_lengths.max() + 1, 45)
for label, color in enumerate(CLASS_COLORS):
    axes[1].hist(review_lengths[y_trainval == label], bins=bins, alpha=0.43,
                 color=color, label=CLASS_NAMES[label])
axes[1].axvline(np.median(review_lengths), color=INK, ls='--',
                label=f'Median: {np.median(review_lengths):.0f}')
axes[1].set(title='Training + validation review lengths', xlabel='Whitespace-separated tokens', ylabel='Reviews')
axes[1].legend(fontsize=9)
plt.tight_layout(); plt.show()

#________________________________________________________________________________________________

# lowercase and turn negations (isn't, can't...) into "not"
def normalize_text(text):
    text = unicodedata.normalize('NFKC', str(text)).replace('’', "'").lower()
    text = re.sub(r"\bcan't\b", 'can not', text)
    text = re.sub(r"\bwon't\b", 'will not', text)
    text = re.sub(r"\bshan't\b", 'shall not', text)
    text = re.sub(r"\bcannot\b", 'can not', text)
    text = re.sub(r"\bca\s+n't\b", 'can not', text)
    text = re.sub(r"\bwo\s+n't\b", 'will not', text)
    text = re.sub(r"n['’]t\b", ' not', text)
    return re.sub(r'\s+', ' ', text).strip()

preprocess_examples = ["I can't recommend this film.", "This isn't good.",
                       "I did not like it.", "It wasn't boring."]
display(pd.DataFrame({'original': preprocess_examples,
                      'normalized': [normalize_text(t) for t in preprocess_examples]}))
# small tests that the function works
assert normalize_text("I can't recommend this film.") == 'i can not recommend this film.'
assert normalize_text("This isn't good.") == 'this is not good.'
assert normalize_text('I can’t recommend this film.') == 'i can not recommend this film.'
assert normalize_text("it did n't work") == 'it did not work'
#________________________________________________________________________________________________

# split the rest into train (75%) and dev (25%)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=SEED,
    stratify= y_trainval,
)
assert len(X_train) == 1050 and len(X_val) == 350 and len(sealed_test) == 600
assert not set(trainval.file) & set(sealed_test.file)
print('Training class counts:', np.bincount(y_train), '| Validation class counts:', np.bincount(y_val))

# plot the three sets
fig, ax = plt.subplots(figsize=(10.5, 2.3))
parts = [('Training', len(X_train), POS), ('Validation', len(X_val), '#8eb7df'),
         ('Sealed test', len(sealed_test), '#efbd9d')]
left = 0
for name, count, color in parts:
    ax.barh(0, count, left=left, height=0.55, color=color, edgecolor='white')
    ax.text(left + count / 2, 0, f'{name}\n{count}', ha='center', va='center',
            color='white' if name == 'Training' else INK, fontweight='bold')
    left += count
ax.set(xlim=(0, 2000), xlabel='Review count', title='Two split levels; one untouched final test')
ax.set_yticks([])
plt.tight_layout(); plt.show()
#________________________________________________________________________________________________

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF features: words (+ word pairs), at most 5000 features
def make_tfidf(ngram_range=(1, 2)):
  
    return TfidfVectorizer(preprocessor=normalize_text, ngram_range=ngram_range,
                           min_df=2, max_features=5000, sublinear_tf=True,
                           stop_words=None, norm='l2')
# check that "not good" becomes one feature
negation_vectorizer = TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2))
negation_vectorizer.fit(["This isn't good.", 'This is good.'])
assert 'not good' in negation_vectorizer.vocabulary_

#________________________________________________________________________________________________


from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# test texts and labels
X_test = sealed_test.text.to_numpy()
y_test = sealed_test.label.to_numpy()

#________________________________________________________________________________________________

# P, R, F1 and PR-AUC for each class, plus their average (macro)
def compute_metrics(model, X, y):
    assert list(model.classes_) == [0, 1]   # column 0 of predict_proba = Negative, column 1 = Positive
    prediction = model.predict(X)           # labels 0/1 for P, R, F1
    proba = model.predict_proba(X)          # probabilities for PR-AUC
    row = {}
    for label, class_name in enumerate(CLASS_NAMES):
        row[f'{class_name} P'] = precision_score(y, prediction, pos_label=label, zero_division=0)
        row[f'{class_name} R'] = recall_score(y, prediction, pos_label=label, zero_division=0)
        row[f'{class_name} F1'] = f1_score(y, prediction, pos_label=label, zero_division=0)
        row[f'{class_name} PR-AUC'] = average_precision_score(y, proba[:, label], pos_label=label)
    for metric in ['P', 'R', 'F1', 'PR-AUC']:
        row[f'Macro {metric}'] = np.mean([row[f'{c} {metric}'] for c in CLASS_NAMES])
    return row

#________________________________________________________________________________________________

# try every C with unigrams and with unigrams+bigrams, score on dev
C_values = [0.01, 0.1, 1, 10, 100]
ngram_options = [(1, 1), (1, 2)]
tuning_rows = []
for ngram_range in ngram_options:
    vec = make_tfidf(ngram_range)
    X_tr = vec.fit_transform(X_train)       # vocabulary and IDF from the training set only
    X_dv = vec.transform(X_val)
    for C in C_values:
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit(X_tr, y_train)
        tuning_rows.append({
            'ngram_range': ngram_range,
            'C': C,
            'Train macro F1': f1_score(y_train, clf.predict(X_tr), average='macro'),
            'Dev macro F1': f1_score(y_val, clf.predict(X_dv), average='macro'),
        })
tuning = pd.DataFrame(tuning_rows)
display(tuning.round(3))
# pick the setting with the best dev macro F1
best = tuning.loc[tuning['Dev macro F1'].idxmax()]
best_ngram, best_C = best['ngram_range'], best['C']
print('Best on dev:', best_ngram, 'C =', best_C, '| dev macro F1 =', round(best['Dev macro F1'], 3))
# plot dev score for each C
fig, ax = plt.subplots(figsize=(7, 4))
for ngram_range in ngram_options:
    part = tuning[tuning['ngram_range'] == ngram_range]
    ax.plot(part['C'], part['Dev macro F1'], marker='o', label=f'n-grams {ngram_range}')
ax.set_xscale('log')
ax.set(xlabel='C = 1/λ (log scale)', ylabel='Dev macro F1', title='Tuning on the development set')
ax.legend(); plt.tight_layout(); plt.show()

#________________________________________________________________________________________________

# train the final models on train with the best settings
final_vectorizer = make_tfidf(best_ngram)
X_train_final = final_vectorizer.fit_transform(X_train)
X_dev_final = final_vectorizer.transform(X_val)
X_test_final = final_vectorizer.transform(X_test)
final_models = {
    'Majority baseline': DummyClassifier(strategy='most_frequent').fit(X_train_final, y_train),
    'Logistic regression': LogisticRegression(C=best_C, max_iter=1000).fit(X_train_final, y_train),
}

#________________________________________________________________________________________________

# one results table for train, dev and test
subsets = {'Train': (X_train_final, y_train),
           'Development': (X_dev_final, y_val),
           'Test': (X_test_final, y_test)}
tables = {}
for subset_name, (X, y) in subsets.items():
    tables[subset_name] = pd.DataFrame(
        {model_name: compute_metrics(model, X, y) for model_name, model in final_models.items()}).T
    print(f'\n=== {subset_name} ===')
    display(tables[subset_name].round(3))

#________________________________________________________________________________________________

# train on 10%, 20%, ..., 100% of train and score on that part and on dev
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
order = np.random.default_rng(SEED).permutation(len(X_train))
model_factories = {
    'Majority baseline': lambda: DummyClassifier(strategy='most_frequent'),
    'Logistic regression': lambda: LogisticRegression(C=best_C, max_iter=1000),
}
curves = {name: {'n': [], 'train': [], 'dev': []} for name in model_factories}
for fraction in fractions:
    n = int(fraction * len(X_train))
    idx = order[:n]
    vec = make_tfidf(best_ngram)
    X_sub = vec.fit_transform(X_train[idx])  # vocabulary only from the subset seen
    X_dv = vec.transform(X_val)
    for name, factory in model_factories.items():
        model = factory().fit(X_sub, y_train[idx])
        curves[name]['n'].append(n)
        curves[name]['train'].append(f1_score(y_train[idx], model.predict(X_sub), average='macro', zero_division=0))
        curves[name]['dev'].append(f1_score(y_val, model.predict(X_dv), average='macro', zero_division=0))

# one plot per model with the two curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (name, c) in zip(axes, curves.items()):
    ax.plot(c['n'], c['train'], marker='o', label='Training subset seen')
    ax.plot(c['n'], c['dev'], marker='o', label='Development set (all)')
    ax.set(title=name, xlabel='Training documents', ylim=(0, 1.05))
    ax.legend(loc='lower right')
axes[0].set_ylabel('Macro F1')
plt.tight_layout(); plt.show()

#________________________________________________________________________________________________
# dataset statistics for the report
stats = pd.DataFrame({
    name: {'Documents': len(texts),
           'Negative': int((labels == 0).sum()),
           'Positive': int((labels == 1).sum()),
           'Mean length (tokens)': np.mean([len(t.split()) for t in texts])}
    for name, (texts, labels) in {'Train': (X_train, y_train),
                                  'Development': (X_val, y_val),
                                  'Test': (X_test, y_test)}.items()}).T
display(stats.round(1))
# vocabulary size
all_words = TfidfVectorizer(preprocessor=normalize_text).fit(X_train)
all_terms = TfidfVectorizer(preprocessor=normalize_text, ngram_range=best_ngram, min_df=2).fit(X_train)
print('Distinct words in train:', len(all_words.vocabulary_))
print(f'Terms {best_ngram} with min_df=2 in train:', len(all_terms.vocabulary_))
print('Features kept (max_features):', len(final_vectorizer.vocabulary_))
