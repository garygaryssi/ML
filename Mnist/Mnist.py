import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import seaborn as sns
import pandas as pd

# 데이터셋 불러오기
def load_dataset(online=False):
    if online:
        (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()

    else:
        path = "./mnist.npz"
        (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data(path)

    return (train_data, train_label), (test_data, test_label)


# 학습을 시키기 위한 함수
def train(tr_data, tr_label):
    clf = RandomForestClassifier()
    clf.fit(tr_data, tr_label)
    print(clf.score(tr_data, tr_label)) # label에 대한 data의 정확도
    dump(clf, "rf_mnist.pkl")



# 테스트 함수
def test(te_data, te_label):
    model = load("rf_mnist.pkl")
    result = model.predict(te_data)

    hit = 0
    miss = 0

    for i in range(len(te_data)):
        if result[i] == te_label[i]:
            hit += 1
        else:
            miss += 1
    print(f'정확도 : {(hit/10000)*100}%, 틀린갯수 : {miss}개')

    return result

# heatmap 작성
def heatmap(result, te_label):
    matrix = []

    # matrix 생성
    for i in range(0, 10):
        matrix.append([0]*10)



    mask = np.zeros_like(matrix, dtype=np.bool)
    for i in range(len(result)):
        matrix[result[i]][te_label[i]] += 1

    for i in range(len(matrix)):
        total = sum(matrix[i])
        for j in range(len(matrix)):
            matrix[i][j] = ((matrix[i][j]) / total * 100)

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                mask[i][j] = True

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, linewidths=3, mask=mask)
    plt.show()



if __name__=="__main__":

    # 데이터셋 불러오기
    (train_data, train_label), (test_data, test_label) = load_dataset()

    # 데이터 전처리
    # 2차원 배열을 1차원 배열로 변환
    train_data = train_data.reshape(60000, 784)

    # 학습함수를 이용한 학습 - 한번 학습한 이후 사용하지 않는다.
    # train(train_data, train_label)

    # 테스트 함수를 이용한 테스트
    test_data = test_data.reshape(10000, 784)
    result = test(test_data, test_label)

    # heatmap
    heatmap(result, test_label)