
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, networkShapes:tuple):
        self.Ws = []
        self.bs = []
        self.sigma = lambda z : 1 / (1 + np.exp(-z))
        self.dSigma = lambda z : np.cosh(z/2)**(-2) / 4
        for i in range(len(networkShapes)):
            shp = networkShapes[i]
            self.Ws.append(np.random.randn(shp[1],shp[0])/2)
            self.bs.append(np.random.randn(shp[1],1)/2)

    def process(self, input):
        immAn = []
        immZ = []
        aVal = input
        for lIdx in range(len(self.Ws)):
            zVal = self.Ws[lIdx] @ aVal + self.bs[lIdx]
            aVal = self.sigma(zVal)
            immAn.append(aVal)
            immZ.append(zVal)

        return aVal, immAn, immZ

    def cost(self,x,y):
        output = self.process(x)
        return np.sum((y - output)**2)/len(output)
    
    def __dAdZ__(self, z):
        return np.cosh(z)**(-2)
    
    def __dCdA__(self,y,an):
        return 2*(an - y)
    
    def train(self,x,y,aggr=3.5,noise=1):
        output,immAn,immZ = self.process(x)
        Ws = []
        bs = []
        invLayerIdx = len(immAn) - 1
        tmp = self.__dCdA__(y,output)
        tmp *= self.dSigma(immZ[invLayerIdx])
        for i in range(invLayerIdx):
            dWs = tmp @ immAn[invLayerIdx-1].T / x.size
            #dWs *= (1+np.random.randn()*noise)
            Ws.append(self.Ws[invLayerIdx] - dWs*aggr)
            dbs = np.sum(tmp, axis=1, keepdims=True) / x.size
            #dbs *= (1+np.random.randn()*noise)
            bs.append(self.bs[invLayerIdx] - dbs*aggr)
            tmp = (tmp.T @ self.Ws[invLayerIdx]).T
            tmp *= self.dSigma(immZ[invLayerIdx-1])
            invLayerIdx -= 1
        dWs = tmp @ x.T / x.size
        dWs *= (1+np.random.randn()*noise)
        Ws.append(self.Ws[invLayerIdx] - dWs*aggr)
        dbs *= (1+np.random.randn()*noise)
        dbs = np.sum(tmp, axis=1, keepdims=True) / x.size
        bs.append(self.bs[invLayerIdx] - dbs*aggr)
        Ws.reverse()
        self.Ws = Ws
        bs.reverse()
        self.bs = bs
        



def lissajous_curve(N=100, a=0.4, b=0.3, q=1.5, c=-1):
    t = np.arange(0,2,2/N)
    y = np.array([a*np.sin(2*np.pi*t),b*np.sin(2*np.pi*q*t+c*np.pi)])
    y = (y+1)/2
    t = np.reshape(t, (1, N))
    return t, y

def nnTraining_graph(NNW:NeuralNetwork, x, y, iterations=10000, aggr=3.5, noise=1, graphStep=100):
    fig,ax = plt.subplots(figsize=(8, 8))
    fig.set_facecolor("slategray")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect(1)
    ax.set_facecolor("lightslategray")
    ax.plot(y[0],y[1], lw=1.5, color="white")

    cycles = int(iterations/graphStep)
    for cycle in range(cycles):
        for i in range(graphStep):
            NNW.train(x, y, aggr=aggr, noise=noise)
        output,_,_ = NNW.process(x)
        # ax.plot(output[0],output[1], lw=2, color=(0.78, 0.89, 0.7, 0.15))
    
    for i in range(iterations-cycles*graphStep):
        NNW.train(x, y, aggr=aggr, noise=noise)

    output,_,_ = NNW.process(x)
    ax.plot(output[0],output[1], lw=2.5, color="tomato")

if __name__ == "__main__":

    import tensorflow as tf

    tfModel = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(units=96, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="sigmoid"),
            tf.keras.layers.Dense(units=32, activation="sigmoid"),
            tf.keras.layers.Dense(units=2, activation="sigmoid")
        ]
    )
    tfModel.summary()

    tfModel.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.007))
    t,y = lissajous_curve()
    tfModel.fit(t.T,y.T,epochs=2000)

    t_n,y_n = lissajous_curve(N=150)
    y_pred = tfModel.predict(t_n.T)
    y_pred = y_pred.T

    fig,ax = plt.subplots(figsize=(8, 8))
    fig.set_facecolor("slategray")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect(1)
    ax.set_facecolor("lightslategray")
    ax.plot(y_n[0],y_n[1], lw=1.5, color="white")
    ax.plot(y_pred[0],y_pred[1], lw=2.5, color="tomato")
    plt.show()

    # Let's test our Neural Network
    # NW = NeuralNetwork([(1,24),(24,32),(32,2)])
    # t,y = lissajous_curve()
    # nnTraining_graph(NW,t,y,iterations=40000,aggr=3.1,noise=0.8,graphStep=100)
    # plt.show()
