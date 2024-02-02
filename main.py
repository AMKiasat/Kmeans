import math

import numpy as np
import random
from scale import scale_into_number
from sklearn.model_selection import train_test_split


def reading_files(filename):
    list1 = []
    list2 = []

    with open(filename, 'r') as file:
        for line in file:
            values = [value.strip("'") for value in line.strip().split(',')]
            if values.__contains__('?'):
                continue
            scaled = scale_into_number(values)
            list2.append(scaled.pop())
            list1.append(scaled)
    data = np.array(list1, dtype=int)
    label = np.array(list2)
    return data, label


def calculate_mean(c, f_num):
    sums = [[0] for _ in range(f_num)]
    x_num = len(c)
    for i in range(len(c)):
        for j in range(len(c[i])):
            sums[j] += (c[i][j] / x_num)
    sums = np.array(sums).flatten()
    # print(sums)
    return sums


def calculate_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def k_means(k, x, offset=0, iteration=10000000):
    feature_num = len(x.T)
    means = np.array(random.sample(list(x), k))
    print(means)
    for _ in range(iteration):
        clusters = [[] for _ in range(k)]
        for i in range(len(x)):
            distances = []
            for j in means:
                distances.append(calculate_distance(j, x[i]))
            d = np.array(distances)
            clusters[np.argmin(d)].append(x[i])
        new_means = []
        for i in range(len(clusters)):
            # print(clusters[i])
            new_means.append(calculate_mean(clusters[i], feature_num))
        same_mean = 0
        new_means = np.array(new_means)
        done = 0
        # print(new_means)
        for i in range(len(new_means)):
            if done == 1:
                break
            for j in range(len(new_means[i])):
                if abs(means[i][j] - new_means[i][j]) > offset:
                    means = new_means
                    done = 1
                    break
        if done == 0:
            return new_means


if __name__ == '__main__':
    data, label = reading_files('Breast Cancer dataset/Breast_Cancer_dataset.txt')
    input_k = int(input("inter your desire k: "))
    # input_k = 2
    # for i in data:
    #     print(i)

    k_mean = k_means(input_k, data)
    print(k_mean)
