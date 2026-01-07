"""
使用的数据集：20news-bydate
使用的基准：TextCNN
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import load_files
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import tarfile
import os
import urllib.request

# --- 数据加载 ---
train_path, test_path = "20news-bydate-train", "20news-bydate-test"

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

# 深度学习预处理参数
MAX_WORDS = 20000          # 词汇表大小
MAX_SEQUENCE_LENGTH = 500  # 序列最大长度
EMBEDDING_DIM = 100        # 嵌入维度

# **步骤 1: 标记化 (对应 CountVectorizer)**
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

# 将文本转换为整数序列
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# **步骤 2: 序列填充 (对应 TfidfTransformer 的统一格式)**
# 填充序列至统一长度，这是 CNN 模型输入所必需的
X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# 标签处理: 转换为独热编码 (One-Hot Encoding)
num_classes = len(newsgroups_train.target_names)
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)


# **步骤 3: 构建 TextCNN 模型 (对应 LinearSVC)**

def create_text_cnn_model(max_words, max_seq_len, embedding_dim, num_classes):
    model = Sequential([
        # 嵌入层：将词汇ID映射为密集的向量
        Embedding(max_words, embedding_dim, input_length=max_seq_len),

        # Conv1D 层：TextCNN 的核心，使用多个过滤器捕获 n-gram 特征
        # 这里我们使用一个简单的 Conv1D，但可以复制此层并使用不同的 kernel_size 来模拟 TextCNN 结构
        Conv1D(filters=128, kernel_size=5, activation='relu'),

        # 池化层：提取最重要的特征（对应 TextCNN 中的 Max-over-time Pooling）
        GlobalMaxPooling1D(),

        # Dropout层：防止过拟合
        Dropout(0.5),

        # 全连接层和输出层
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# 创建并打印模型摘要
model = create_text_cnn_model(MAX_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, num_classes)
print(model.summary())

# **步骤 4: 模型训练 (对应 text_clf.fit())**
print("\n--- 开始训练 TextCNN 模型 ---")
history = model.fit(
    X_train_padded,
    y_train_one_hot,
    epochs=5,  # 深度学习通常需要多轮训练
    batch_size=32,
    validation_data=(X_test_padded, y_test_one_hot)
)

# **步骤 5: 模型预测 (对应 text_clf.predict())**
# 预测返回的是概率，需要转换为类别索引
print("\n--- 开始模型预测 ---")
y_pred_probs = model.predict(X_test_padded)
y_predicted_classes = np.argmax(y_pred_probs, axis=1)

# **步骤 6: 评估 (对应 print(metrics.classification_report()))**
print("\n--- 分类报告 ---")
print(metrics.classification_report(
    y_test,
    y_predicted_classes,
    target_names=newsgroups_test.target_names,
    digits=4
))