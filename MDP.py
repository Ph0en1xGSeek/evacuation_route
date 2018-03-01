'''
更新策略：策略为向周围最大q值的动作a，每次更新v后马上就改变策略
'''

import numpy as np
import pandas as pd
import time
import json
from sklearn.preprocessing import MinMaxScaler
from time import time

N_STATES = 58
Coord = None
danger = None  # 评价值
dis = None
TERMINATE_STATE = None
N_TERMINATE = 5
Reward = None
value = None
policy = []
N_Neighbor = 5
knn = None
GAMMA = 0.9  # 奖励递减值
MAX_ITERATION = 200


# 策略评估
def updateV():
    tmpV = np.zeros([N_STATES])
    for i in range(N_STATES):
        cnt = 0
        for j in policy[i]:
            cnt += 1
            tmpV[i] = (cnt - 1) / cnt * tmpV[i] + (Reward[i, j] + GAMMA * value[j]) / cnt
    return tmpV


# 策略改进
def updatePi(value):
    '''
    select the k nearest places and pick the max value one as policy
    :return:
    '''
    for i in range(N_STATES):
        mx = -99999999.0
        for j in knn[i]:

            if Reward[i, j] + GAMMA * value[j] > mx:
                mx = Reward[i, j] + GAMMA * value[j]
                policy[i] = [j]
            elif value[j] == mx:
                policy[i].append(j)
    return


if __name__ == '__main__':
    with open('data2.json') as f:
        dic = f.read()
        f.close()
    dic = json.loads(dic)
    N_STATES = len(dic)
    Coord = np.zeros([N_STATES, 2])
    danger = np.zeros([N_STATES, 1])
    value = np.zeros([N_STATES])
    for i in range(N_STATES):
        policy.append([])
    for i, item in enumerate(dic):
        Coord[i, 0] = item['lng']
        Coord[i, 1] = item['lat']
        danger[i] = item['count']

    TERMINATE_STATE = danger.argsort(axis=0)[:N_TERMINATE]
    # danger score
    danger = np.tile(danger, (1, N_STATES))
    danger = danger - danger.T
    # distance
    sum_X = np.sum(np.square(Coord), 1)
    dis = np.sqrt(np.add(np.add(-2 * np.dot(Coord, Coord.T), sum_X).T, sum_X))  # 欧氏距离矩阵
    knn = dis.argsort()[:, 1:N_Neighbor + 1]
    # print(knn)
    # scale distance
    dismax = dis.max()
    dismin = dis.min()

    dis = (dis - dismin) / (dismax - dismin)

    # scale danger
    dangermax = danger.max()
    dangermin = danger.min()
    danger = danger / dangermax
    # print(danger)

    r = danger / np.power(dis, danger / np.abs(danger))
    # r = np.exp(danger) - np.exp(dis)
    # reward function
    r[range(N_STATES), range(N_STATES)] = -1
    Reward = r

    for i in range(MAX_ITERATION):
        updatePi(value)
        tmpvalue = updateV()
        if (tmpvalue == value).all():
            break
        else:
            value = tmpvalue
        print("iter: {}".format(i))
        print("value", value)
    print("policy: ", policy)

    line_data = []
    for i in range(N_STATES):
        if i in TERMINATE_STATE:
            continue
        for j in policy[i]:
            road_line = {}
            lineStr = '{},{};'.format(Coord[i, 0], Coord[i, 1])
            lineStr += '{},{}'.format(Coord[j, 0], Coord[j, 1])
            road_line['ROAD_LINE'] = lineStr
            line_data.append(road_line)

    jsonStr = json.dumps(line_data)
    with open('line_data_MDP.json', 'w') as file:
        print(jsonStr, file=file)
        file.close()

    np.savetxt('value.txt', value)
    np.savetxt('knn.txt', knn, fmt="%d")