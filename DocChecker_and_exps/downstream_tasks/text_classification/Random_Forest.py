"""
使用的数据集：20news-bydate
使用的基准：Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_files
import tarfile
import urllib.request
import os


# --- 修正后的下载函数 ---
def download_20newsgroups_by_date():
    # 采用包含预分好 train/test 集的压缩包 URL
    url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"

    # 文件名和主目录名
    archive_filename = "20news-bydate.tar.gz"
    extract_dir = "20news-bydate"

    # 1. 下载文件
    if not os.path.exists(archive_filename):
        print("Downloading dataset...")
        try:
            # 使用 tempfile 来避免直接写入，更安全
            urllib.request.urlretrieve(url, archive_filename)
        except Exception as e:
            print(f"Error downloading file: {e}")
            print("Please manually download the file and place it in the script directory.")
            return None, None  # 返回 None 表示失败

    # 2. 解压文件
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        try:
            with tarfile.open(archive_filename, 'r:gz') as tar:
                # 检查 tar 文件内容
                # tar.extractall() 会创建 '20news-bydate-train' 和 '20news-bydate-test'
                # 这两个文件夹位于当前工作目录
                tar.extractall()
        except tarfile.ReadError:
            print("Error: The downloaded file is corrupt or not a valid tar.gz file.")
            return None, None

    # 3. 返回正确的路径 (注意：它们现在位于当前目录下，而不是嵌套在 '20news-bydate' 中)
    # 原始的 20news-bydate.tar.gz 解压后会得到两个文件夹：
    # 20news-bydate-train 和 20news-bydate-test
    return "20news-bydate-train", "20news-bydate-test"


# 使用修正后的下载逻辑
# train_path, test_path = download_20newsgroups_by_date()
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

# ... [Pipeline code remains the same] ...
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=100)),
                     ])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted, target_names=newsgroups_test.target_names))