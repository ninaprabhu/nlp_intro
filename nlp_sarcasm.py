import pandas as pd, numpy as np, re, time
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = pd.read_json("archive/Sarcasm_Headlines_Dataset.json", lines=True)
if (data.isnull().any(axis=0).any()):
    raise ValueError("Data must be cleaned.")

# Relacing special symbols and digits in headline column
# re stands for Regular Expression
data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))

# getting features and labels
features = data['headline']
labels = data['is_sarcastic']

# Stemming our data
ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ‘ ‘.join([ps.stem(word) for word in x]))

