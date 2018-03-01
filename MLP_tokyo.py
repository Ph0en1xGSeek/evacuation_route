import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
import json

n_1 = 40
n_2 = 20
n_3 = 10
startplace = 45
dataset = 'tokyo'
dataset2 = 'tokyo'

training_epochs = 300
new_epochs = 0

def score(arr1, arr2):
    scalar = prep.MinMaxScaler()
    dis1 = arr1.reshape(-1, 1)
    dis2 = arr2.reshape(-1, 1)
    dis1 = scalar.fit_transform(dis1).reshape(-1)
    dis2 = scalar.fit_transform(dis2).reshape(-1)
    loss = np.sum((dis1-dis2) ** 2, 0)
    return dis1, dis2, loss

def score2(arr1, arr2):
    scalar = prep.StandardScaler()
    dis1 = arr1.reshape(-1, 1)
    dis2 = arr2.reshape(-1, 1)
    dis1 = scalar.fit_transform(dis1).reshape(-1)
    dis2 = scalar.fit_transform(dis2).reshape(-1)
    loss = np.sum((dis1 - dis2) ** 2, 0)
    return dis1, dis2, loss


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32, seed=1)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, optimizer = tf.train.AdamOptimizer(), scale = 0.01):
        #n_input:输入变量数
        #n_hidden:隐藏层节点数
        #transfer_function:隐藏层激活函数
        #optimizer:优化器Adam
        #scale:高斯噪声系数

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale

        # network structure
        # distribution on the hidden layer relies on the activation function
        self.x = tf.placeholder(tf.float32, [None, self.n_input], name='X')
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # encode
        # self.att1 = self.x * self.weights['attention1']
        # self.tmp = tf.nn.softmax(self.x)
        self.layer1 = tf.nn.sigmoid(tf.add(tf.matmul(self.x , self.weights['w1']), self.weights['b1']))
        self.layer2 = tf.nn.softplus(tf.add(tf.matmul(self.layer1, self.weights['w2']), self.weights['b2']))
        self.layer3 = tf.nn.softplus(tf.add(tf.matmul(self.layer2, self.weights['w3']), self.weights['b3']))

        self.hidden = tf.nn.softplus(tf.add(tf.matmul(self.layer3, self.weights['w4']), self.weights['b4']))

        # decode
        self.layer5 = tf.nn.softplus(tf.add(tf.matmul(self.hidden, self.weights['w5']), self.weights['b5']))
        self.layer6 = tf.nn.softplus(tf.add(tf.matmul(self.layer5, self.weights['w6']), self.weights['b6']))
        self.layer7 = tf.nn.sigmoid(tf.add(tf.matmul(self.layer6, self.weights['w7']), self.weights['b7']))

        self.reconstruction = tf.nn.tanh(tf.add(tf.matmul(self.layer7, self.weights['w8']), self.weights['b8']))
        # self.reconstruction = (self.att2 * (1.0/self.weights['attention2']))

        # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.cost = tf.reduce_sum(-tf.reduce_sum(tf.nn.softmax(self.x) * tf.log(tf.nn.softmax(self.reconstruction)), reduction_indices=[1]))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        self.sess.run(init)
        # self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)

    def _initialize_weights(self):
        all_weights = {}
        # all_weights['attention1'] = (tf.Variable(tf.ones([self.n_input], dtype=tf.float32), name='attention1'))
        # all_weights['attention2'] = (tf.Variable(tf.ones([self.n_input], dtype=tf.float32)))

        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_1), name='w1')
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_1], dtype=tf.float32), name='b1')
        all_weights['w2'] = tf.Variable(xavier_init(self.n_1, self.n_2), name='w2')
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_2], dtype=tf.float32), name='b2')
        all_weights['w3'] = tf.Variable(xavier_init(self.n_2, self.n_3), name='w3')
        all_weights['b3'] = tf.Variable(tf.zeros([self.n_3], dtype=tf.float32), name='b3')
        all_weights['w4'] = tf.Variable(xavier_init(self.n_3, self.n_hidden), name='w4')
        all_weights['b4'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='b4')

        all_weights['w5'] = tf.Variable(xavier_init(self.n_hidden, self.n_3), name='w5')
        all_weights['b5'] = tf.Variable(tf.zeros([self.n_3], dtype=tf.float32), name='b5')
        all_weights['w6'] = tf.Variable(xavier_init(self.n_3, self.n_2), name='w6')
        all_weights['b6'] = tf.Variable(tf.zeros([self.n_2], dtype=tf.float32), name='b6')
        all_weights['w7'] = tf.Variable(xavier_init(self.n_2, self.n_1), name='w7')
        all_weights['b7'] = tf.Variable(tf.zeros([self.n_1], dtype=tf.float32), name='b7')
        all_weights['w8'] = tf.Variable(xavier_init(self.n_1, self.n_input), name='w8')
        all_weights['b8'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='b8')
        return all_weights

    def partial_fit(self, X):
        #get the value of cost and optimize it
        cost, opt = self.sess.run((self.cost, self.optimizer),
                             feed_dict = {self.x : X, self.scale : self.training_scale})
        return cost

    def calc_total_cost(self, X):
        #get the value of cost
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale : self.training_scale})

    def transform(self, X):
        #get the value of hidden layer
        return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale:self.training_scale})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size =  self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden : hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x : X, self.scale : self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    # def getAttent1(self):
        # return self.sess.run(self.weights['attention1'])

    # def getAttent2(self):
        # return self.sess.run(self.weights['attention2'])

def standard_scale(X_train, X_test):
    #标准化数据，均值为0，标准差为1
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    preprocessor = prep.StandardScaler().fit(X_test)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    #get a batch size of batch_size randomly
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index : (start_index + batch_size)]

def main(_):
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # print (mnist.test.labels)
    # X_train = mnist.train.images
    # X_test = mnist.test.images
    # labels = mnist.test.labels

    X_train = np.loadtxt("data/"+dataset+"_X.txt");
    X_test = np.loadtxt("data/"+dataset2+"_X.txt")
    try:
        labels = np.loadtxt("data/"+dataset+"_labels.txt");
    except IOError:
        labels = np.zeros(X_test.shape[0])
        labels[startplace] = 1

    try:
        names = open("data/"+dataset+"_names.txt", 'r', encoding='utf8').read();
        names = names.split()
    except IOError:
        names = np.array(range(X_test.shape[0])).astype(dtype=str)


    n, m = X_train.shape


    # Scale
    X_train, X_test = standard_scale(X_train, X_test)

    print('calculating PCA...')
    pca = PCA(n_components=2)
    Y_pca = pca.fit_transform(X_test)

    print('calculating MDS...')
    mds = manifold.MDS(n_components=2, max_iter=100, n_init=1)
    Y_mds = mds.fit_transform(X_test)

    print('calculating ISOMAP...')
    isomap = manifold.Isomap(n_neighbors=5, n_components=2)
    Y_iso = isomap.fit_transform(X_test)

    print('calculating TSNE...')
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y_tsne = tsne.fit_transform(X_test)

    #===============================================================Our Method==================


    n_samples = X_train.shape[0]
    batch_size = n_samples
    display_step = 1

    autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=m,
                                                   n_hidden=2,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5),
                                                   scale=0.005)
    print('calculating AE...')
    total_batch = int(n_samples // batch_size)
    for epoch in range(training_epochs):
        avg_cost = 0.
        for i in range(total_batch+1):
            # batch_xs = get_random_block_from_data(X_train, batch_size)
            if i+batch_size < n_samples:
                batch_xs = X_train[i : i+batch_size]
            else:
                batch_xs = X_train[i: ]
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples
        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch + 1), "cost=", '{:.9f}'.format(avg_cost))

    for epoch in range(new_epochs):
        avg_cost = 0.
        for i in range(total_batch + 1):
            # batch_xs = get_random_block_from_data(X_train, batch_size)
            if i + batch_size < n_samples:
                batch_xs = X_test[i: i + batch_size]
            else:
                batch_xs = X_test[i:]
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples

        if epoch % display_step == 0:
            print('Epoch:', '%04d'%(epoch+1), "cost=", '{:.9f}'.format(avg_cost))
    # print('Total cost: ' + str(autoencoder.calc_total_cost(X_test)))
    Y = autoencoder.transform(X_test)

    dif1 = (X_test-X_test[startplace])**2
    dif2 = (Y - Y[startplace]) ** 2
    dif3 = (Y_pca - Y_pca[startplace]) ** 2
    dif4 = (Y_mds - Y_mds[startplace]) ** 2
    dif5 = (Y_iso - Y_iso[startplace]) ** 2
    dif6 = (Y_tsne - Y_tsne[startplace]) ** 2

    dis1 = np.sqrt(np.sum(dif1, 1))
    dis2 = np.sqrt(np.sum(dif2, 1))
    dis3 = np.sqrt(np.sum(dif3, 1))
    dis4 = np.sqrt(np.sum(dif4, 1))
    dis5 = np.sqrt(np.sum(dif5, 1))
    dis6 = np.sqrt(np.sum(dif6, 1))

    # rank1 = np.argsort(dis1)
    # rank1 = {rank: idx for idx, rank in enumerate(rank1)}
    # rank2 = np.argsort(dis2)
    # rank2 = {rank: idx for idx, rank in enumerate(rank2)}
    # print(rank1)
    # print(rank2)

    # _, _, loss = score2(dis1, dis2)
    _, dis3, loss = score(dis1, dis3)
    print("The loss of PCA is: ", loss)
    _, dis4, loss = score(dis1, dis4)
    print("The loss of MDS is: ", loss)
    _, dis5, loss = score(dis1, dis5)
    print("The loss of ISOMAP is: ", loss)
    _, dis6, loss = score(dis1, dis6)
    print("The loss of TSNE is: ", loss)
    dis1, dis2, loss = score(dis1, dis2)
    print("The loss of AE is: ", loss)

    with open('data/tokyo_gps.txt') as f:
        X = f.read()
        X = X.split('\n')
        f.close()
    mapDic = []
    for i, name in enumerate(names):
        X[i] = X[i].split(' ')
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (1 - dis1[i])
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data1.json", "w")
    print(jsonStr, file=f)
    f.close()

    mapDic = []
    for i, name in enumerate(names):
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (1 - dis2[i])
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data2.json", "w")
    print(jsonStr, file=f)
    f.close()

    mapDic = []
    for i, name in enumerate(names):
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (1 - dis3[i])
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data3.json", "w")
    print(jsonStr, file=f)
    f.close()

    mapDic = []
    for i, name in enumerate(names):
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (np.fabs(dis1[i]-dis2[i]))
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data4.json", "w")
    print(jsonStr, file=f)
    f.close()

    mapDic = []
    for i, name in enumerate(names):
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (np.fabs(dis1[i]-dis3[i]))
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data5.json", "w")
    print(jsonStr, file=f)
    f.close()

    mapDic = []
    for i, name in enumerate(names):
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (1 - dis4[i])
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data6.json", "w")
    print(jsonStr, file=f)
    f.close()

    mapDic = []
    for i, name in enumerate(names):
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (1 - dis5[i])
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data7.json", "w")
    print(jsonStr, file=f)
    f.close()

    mapDic = []
    for i, name in enumerate(names):
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (np.fabs(dis1[i] - dis4[i]))
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data8.json", "w")
    print(jsonStr, file=f)
    f.close()

    mapDic = []
    for i, name in enumerate(names):
        tmpdic = {}
        tmpdic["lng"] = float(X[i][0])
        tmpdic["lat"] = float(X[i][1])
        tmpdic["count"] = 100 * (np.fabs(dis1[i] - dis5[i]))
        mapDic.append(tmpdic)
    jsonStr = json.dumps(mapDic)
    f = open("data9.json", "w")
    print(jsonStr, file=f)
    f.close()

    # fig = plt.figure(figsize=(15, 8))
    # plt.suptitle(dataset)
    # ax = fig.add_subplot(1, 2, 1)
    # plt.title('MLP')
    # plt.scatter(Y[:,0], Y[:,1], c=labels, cmap=plt.cm.Spectral, marker='.')
    # ax = fig.add_subplot(1, 2, 2)
    # plt.title('pca')
    # plt.scatter(Y_pca[:,0], Y_pca[:,1], c=dis2, cmap=plt.cm.Spectral)

    fig, ax = plt.subplots()
    sc = plt.scatter(Y[:,0], Y[:,1], marker='o', c=dis4, cmap=plt.cm.Spectral)
    plt.xlim(xmin=0, xmax=8)  # adjust the max leaving min unchanged
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = names[ind["ind"][0]]
        annot.set_text(text)
        # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        # annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    # plt.figure()
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # Y3 = tsne.fit_transform(X_test)
    # plt.scatter(Y3[:, 0], Y3[:, 1], s=(10 - labels) ** 2, c=labels, cmap=plt.cm.Spectral)



    plt.show()




if __name__ == '__main__':
    tf.app.run(main = main)
