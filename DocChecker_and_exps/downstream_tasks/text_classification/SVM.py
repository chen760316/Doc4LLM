"""
使用的数据集：20news-bydate
使用的基准：SVM
"""
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_files
import tarfile
import urllib.request
import os

# 使用修正后的下载逻辑
train_path, test_path = "20news-bydate-train", "20news-bydate-test"

# 如果下载或解压失败，则退出
if train_path is None:
    exit()

# --- 数据加载和分类器 ---
try:
    newsgroups_train = load_files(train_path, encoding='latin-1')
    newsgroups_test = load_files(test_path, encoding='latin-1')
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Could not find the extracted dataset folders at {e.filename}")
    print("Please verify that the tar.gz file was successfully extracted, and that the folders '20news-bydate-train' and '20news-bydate-test' exist in the same directory as this script.")
    exit()

X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))