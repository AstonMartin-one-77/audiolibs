
import numpy as np

class BaseLayer:
    def __init__(self, units:int):
        self.units = units
        self.W = None
        self.b = None
        self.bLayer = None
        self.fLayer = None
        self.optimizer = None

    def getUnits(self):
        return self.units

    def backward(self, d:np.array):
        pass

    def forward(self, aVal:np.array):
        pass

    def compile(self):
        pass

class BaseOptimizer:
    def __init__(self, lrate:float = 0.001):
        self.lrate = lrate
    
    def dW(self, g:np.array):
        return g*self.lrate
    
    def db(self, g:np.array):
        return np.sum(g, axis=1, keepdims=True)*self.lrate

class InputLayer(BaseLayer):
    def __init__(self, units: int):
        super().__init__(units)

    def setFLink(self, fLayer:BaseLayer):
        self.fLayer = fLayer

    def forward(self, aVal:np.array):
        return self.fLayer.forward(aVal)
    

class DenseLayer(BaseLayer):
    def __init__(self, units:int, type:str = "sigmoid", optimizer:BaseOptimizer = BaseOptimizer(0.031)):
        super().__init__(units=units)
        if "sigmoid" == type:
            self.G = lambda z : 1 / (1 + np.exp(-z))
            self.dG = lambda z : np.cosh(z/2)**(-2) / 4
        elif "relu" == type:
            self.G = lambda z : np.maximum(0, z)
            self.dG = lambda z : np.where(z > 0, 1, 0)
        else:
            self.G = lambda z : z
            self.dG = lambda z : 1
        self.optimizer = optimizer


    # Input of func: 2*(y - a(n)) where a(n) = G(z(n)), z(n) = W(n) @ a(n-1) + b(n)
    #       dCdA: 2*(diff), where diff = y - a(n)
    # 1 step: dAdZ - activation dependent partial derivative
    #       1.a: dZdW => a(n-1) & dZdB => 1 (for internal Wn/b update of current layer)
    # 2 step: dZdA: W(n) (input for previous layer - backprop) (recursive)
    def backward(self, d:np.array):
        dAdZ = d*self.dG(self.saveZ)
        dZdA = dAdZ.T @ self.W
        self.W += self.optimizer.dW(dAdZ @ self.prevA.T)
        self.b += self.optimizer.db(dAdZ)
        self.bLayer.backward(dZdA.T)

    def forward(self, aVal:np.array):
        self.prevA = aVal
        self.saveZ = self.W @ aVal + self.b
        output = self.G(self.saveZ)
        if self.fLayer is not None:
            return self.fLayer.forward(output)
        return output

    def setBLink(self, bLayer:BaseLayer):
        self.bLayer = bLayer
    
    def setFLink(self, fLayer:BaseLayer):
        self.fLayer = fLayer
    
    def setOptimizer(self, optimizer:BaseOptimizer):
        self.optimizer = optimizer

    def compile(self, W:np.array = None, b:np.array = None):
        if W is not None:
            self.W = W
        else:
            self.W = np.random.randn(self.units, self.bLayer.getUnits()) / 2
        if b is not None:
            self.b = b
        else:
            self.b = np.random.randn(self.units, 1) / 2

class AdamOptimizer(BaseOptimizer):
    def __init__(self, lrate:float = 0.001, b1:float = 0.9, b2:float = 0.999):
        super().__init__(lrate=lrate)
        self.b1 = b1
        self.b2 = b2
        self.m = 0
        self.v = 0
        self.e = 0.001
        self.i = 1
    
    # 1 step: Moving average of gradient (first estimator)
    # 2 step: Moving of squared gradient (second estimator)
    # 3 step: Bias correction for first estimator
    # 4 step: Bias correction for second estimator
    # 5 step: Calculate weights addiction
    def dW(self, g:np.array):
        self.m = self.b1 * self.m + (1 - self.b1)*g
        self.v = self.b2 *self.v + (1 - self.b2)*(g**2)
        m_hat = self.m / (1 - np.power(self.b1, self.i))
        v_hat = self.v / (1 - np.power(self.b2, self.i))
        self.i += 1
        return self.lrate * m_hat / (np.sqrt(v_hat) + self.e)

class NeuralNetwork_flow:
    def __init__(self, layersSequence:list):
        assert type(layersSequence[0]) == InputLayer
        self.layers = layersSequence
        self.inLayer = prevLayer = self.layers[0] # Input layer
        for i in range(1, len(self.layers)):
            prevLayer.setFLink(self.layers[i])
            self.layers[i].setBLink(prevLayer)
            prevLayer = self.layers[i]
        self.outLayer = self.layers[-1]
        for i in range(1, len(self.layers)):
            self.layers[i].compile()

    def train(self, x:np.array, y:np.array, iterations=10000):
        for i in range(iterations):
            itrRes = self.inLayer.forward(x)
            self.outLayer.backward(2*(y-itrRes))
    
    def error(self, x, y):
        return np.sum((self.inLayer.forward(x)-y)**2)/len(y)
    
    def predict(self, x):
        return self.inLayer.forward(x)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import NeuralNetwork_generic as NNg

    # input = InputLayer(units=1)
    # dl1 = DenseLayer(units=24)
    # input.setFLink(dl1)
    # dl1.setBLink(input)
    # dl2 = DenseLayer(units=32)
    # dl1.setFLink(dl2)
    # dl2.setBLink(dl1)
    # output = DenseLayer(units=2)
    # dl2.setFLink(output)
    # output.setBLink(dl2)
    # dl1.setOptimizer(AdamOptimizer(lrate=0.0031))
    # dl2.setOptimizer(AdamOptimizer(lrate=0.0031))
    # output.setOptimizer(AdamOptimizer(lrate=0.0031))

    # NW = NeuralNetwork([(1,5),(5,4),(4,2)])
    # w0 = np.array([[ 0.06332294],       [-1.64162638],       [ 0.13457888],       [ 0.4365871 ],       [ 0.79491503]])
    # b0 = np.array([[-0.13764146],       [ 0.11569728],       [-0.51257726],       [-0.31815902],       [-0.15085569]])
    # w1 = np.array([[ 0.54963372, -0.33513557, -0.24486069,  0.12902891, -0.4986361 ],       [ 0.03465819,  0.01338351, -0.10258007,  0.86133015,  0.73636143],       [ 0.03382981,  0.19389837, -0.02179771, -0.34117452, -0.37477451],       [-0.79792423,  0.86582359,  0.25982899, -0.25303926,  0.44011548]])
    # b1 = np.array([[-0.64159605],       [ 0.25569778],       [-0.37210437],       [-0.0923177 ]])
    # w2 = np.array([[ 0.73193099,  0.11739459, -0.7096986 ,  0.45298538],       [ 0.28562191,  0.24134889, -0.0294186 , -0.61084058]])
    # b2 = np.array([[-0.28801579],       [ 0.11429052]])
    # dl1.compile()
    # dl2.compile()
    # output.compile()

    NNf = NeuralNetwork_flow([
                InputLayer(units=1),
                DenseLayer(units=24, type="linear", optimizer=AdamOptimizer(0.005)),
                DenseLayer(units=32, type="sigmoid", optimizer=AdamOptimizer(0.0031)),
                DenseLayer(units=2)])

    t,y = NNg.lissajous_curve()

    print(NNf.error(t, y))

    NNf.train(t, y, iterations=40000)

    print(NNf.error(t, y))

    t_n,y_n = NNg.lissajous_curve(N=150)
    y_pred = NNf.predict(t_n)

    fig,ax = plt.subplots(figsize=(8, 8))
    fig.set_facecolor("slategray")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect(1)
    ax.set_facecolor("lightslategray")
    ax.plot(y_n[0],y_n[1], lw=1.5, color="white")
    ax.plot(y_pred[0],y_pred[1], lw=2.5, color="tomato")
    plt.show()
