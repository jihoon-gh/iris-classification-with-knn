import numpy as np


class KNN:

    K = 0   # main.py에서 입력받은 K.
    features = np.empty((0, 4), int)  # iris data
    target = np.empty((0, 4), int)  #target -> class
    test = np.empty((0, 4), int)    # main.py에서 구한 test data list 중 이번에 test할 data (매 15번째 데이터를 numpy array로 만든 것)
    test_indexes = []   # 테스트 데이터 전체
    dist = {} # 계산된 최단거리를 index : distance 형로 저장할 dictionary
    calculated_dist = []    # dictionary를 sorted를 이용해 정렬했을 시 나오는 tuple list 저장
    result_mv = []    # test case에 대해 majority vote를 실행하여 얻은 class를 저장
    result_wmv = []    # test case에 대해 weight majority vote를 실행하여 얻은 class를 저장

    #  생성자, input feature와 target data를 parameter로 받음
    def __init__(self, k, x, target):
        self.K = k
        self.features = x
        self.target = target
        self.make_test()

    # features에서 매 15번째 인덱스를 뽑아내어 test data numpy array 만듦
    def make_test(self):
        temp = 14  # initial index of test data. 14, 29, 44....
        for i in range(len(self.features)):
            if i == temp:
                self.test = np.append(self.test, np.array([self.features[i]]), axis=0)
                self.test_indexes.append(i)
                temp = temp + 15

    #   모든 test data에 대해 계산 method를 실행하고 이를 통해 classfication 결과를 도출해 저장.
    def make_result(self):
        for i in range(len(self.test)):
            self.dist = {}
            self.calculated_dist.clear()
            self.calculate_distance(i)
            self.get_k_nearest_neighbors()
            self.get_majority_vote()
            self.get_weighted_majority_vote()

    # 모든 데이터에 대하여 test data와의 최단 거리를 구하는 method. Euclidean distance 이용
    # 전체 data에서 매 15번 째 data는 test data임으로 이를 제외해 줌
    def calculate_distance(self, n):
        num = 14
        for i in range(len(self.features)):
            if i == num:
                num += 15
                continue
            temp = self.features[i]-self.test[n]
            temp **= 2
            res = np.sqrt(np.sum(temp))
            self.dist[i] = res

    # dist dictionary에 저장된 값을 정렬하고, tuple의 list 형태에서 k개만 남겨서 slice 후 저장
    def get_k_nearest_neighbors(self):
        self.calculated_dist = sorted(self.dist.items(), key=lambda item: item[1])
        self.calculated_dist = self.calculated_dist[:self.K]

    # class의 빈도수를 카운팅하고, test의 class가 될 max값의 index를 result_mv에 저장
    def get_majority_vote(self):
        count = [0, 0, 0]
        for j in self.calculated_dist:
            count[self.target[j[0]]] += 1

        num = count.index(max(count))
        self.result_mv.append(num)

    # 계산된 최단 거리에 대해 1/d 의 가중치를 곱하여 classfication 후 result_wmv에 결과 저장
    def get_weighted_majority_vote(self):
        count = [0, 0, 0]
        for j in self.calculated_dist:
            count[self.target[j[0]]] += 1/j[1]

        num = count.index(max(count))
        self.result_wmv.append(num)

    # 모든 test data 에 대해 majorty vote를 통해 clssfication한 결과 list를 반환.
    def get_result_mv(self):
        return self.result_mv

    # 모든 test data 에 대해 weighted majorty vote를 통해 clssfication한 결과 list를 반환.
    def get_result_wmv(self):
        return self.result_wmv

    def get_test_indexes(self):
        return self.test_indexes







