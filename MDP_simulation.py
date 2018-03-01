import numpy as np
import json

startplace = 49
N_LINE = 50
MAX_LEN = 6
TERMINATE_STATE = None
N_TERMINATE = 1
np.random.seed(0)
with open('data2.json') as f:
    dic = f.read()
    f.close()
dic = json.loads(dic)
N_STATES = len(dic)
Coord = np.zeros([N_STATES, 2])
danger = np.zeros([N_STATES, 1])
for i, item in enumerate(dic):
    Coord[i, 0] = item['lng']
    Coord[i, 1] = item['lat']
    danger[i] = item['count']

TERMINATE_STATE = danger.argsort(axis=0)[:N_TERMINATE]
# danger score
# danger = np.tile(danger, (1, N_STATES))
# danger = danger - danger.T

value = np.loadtxt('value.txt')
knn = np.loadtxt('knn.txt', dtype=np.int32)

line_data = [] #路线

for i in range(N_LINE):
    cur = startplace
    road_line = {}
    lineStr = ''
    for j in range(MAX_LEN):
        lineStr += '{},{};'.format(Coord[cur, 0], Coord[cur, 1])
        probability = np.array([np.exp(value[a]) for a in knn[cur]])
        probability = probability / np.sum(probability)
        cur = np.random.choice(knn[cur], p=probability)
        if cur in TERMINATE_STATE:
            break
    lineStr += '{},{}'.format(Coord[cur, 0], Coord[cur, 1])
    road_line['ROAD_LINE'] = lineStr
    line_data.append(road_line)

jsonStr = json.dumps(line_data)
with open('line_data_MDP2.json', 'w') as file:
    print(jsonStr, file=file)
    file.close()