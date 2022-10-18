import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from KNN import KNN
iris = load_iris()

X = iris.data  # data input
y = iris.target  # target data (label)
y_names = iris.target_names # name of class

K = int(input())    # value of K.

knn = KNN(K, X, y)  # 객체 생성
knn.make_result()   # KNN class의 method를 이용하여 거리 계산 등 수행
mv = knn.get_result_mv()    # majority vote를 실행하여 얻은 결과 리스트
wmv = knn.get_result_wmv()  # weight majority vote를 실행하여 얻은 결과 리스트
test_index = knn.get_test_indexes() # knn 클래스 내부에서 test data의 인덱스 리스트

print("\nresult of majority_vote")  # weighted majority vote의 결과 출력
for i in range(len(mv)):
    print("Test index: %d, Computed class: %s, True class: %s" % (i, y_names[mv[i]], y_names[y[test_index[i]]]))

print("\nresult of weighted_majority_vote")     # weighted majority vote의 결과 출력
for i in range(len(wmv)):
    print("Test index: %d, Computed class: %s, True class: %s" % (i, y_names[wmv[i]], y_names[y[test_index[i]]]))


