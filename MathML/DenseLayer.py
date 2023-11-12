
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
        return -g*self.lrate
    
    def db(self, g:np.array):
        return -np.sum(g, axis=1, keepdims=True)*self.lrate

class InputLayer(BaseLayer):
    def __init__(self, units: int):
        super().__init__(units)

    def setFLink(self, fLayer:BaseLayer):
        self.fLayer = fLayer

    def forward(self, aVal:np.array):
        return self.fLayer.forward(aVal)
    

class DenseLayer(BaseLayer):
    def __init__(self, units:int, type:str = "sigmoid"):
        super().__init__(units=units)
        if "sigmoid" == type:
            self.G = lambda z : 1 / (1 + np.exp(-z))
            self.dG = lambda z : np.cosh(z/2)**(-2) / 4
        elif "relu" == type:
            self.G = lambda z : np.maximum(0, z)
            self.dG = lambda z : z
        else:
            self.G = lambda z : z
            self.dG = lambda z : z

    # 1) (y - a(n))**2 where a(n) = G(z(n)), z(n) = W(n) @ a(n-1) + b(n)
    #       dCdA: 2*(diff), where diff = y - a(n)
    #       dAdZ: activation dependent
    #           dZdW: a(n-1) & dZdB: 1 (for internal update of current layer)
    #       dZdA: W(n)
    #       dAdZ: activation dependent
    def backward(self, d:np.array):
        dAdZ = d*self.dG(self.saveZ)
        dZdA = dAdZ.T @ self.W
        # self.W += self.optimizer.dW(dAdZ @ self.prevA.T)
        dW = dAdZ @ self.prevA.T / 100
        self.W -= dW * 3.1
        # self.b += self.optimizer.db(dAdZ)
        db = np.sum(dAdZ, axis=1, keepdims=True) / 100
        self.b -= db * 3.1
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

    def compile(self):
        self.W = np.random.randn(self.units, self.bLayer.getUnits()) / 2
        self.b = np.random.randn(self.units, 1) / 2

class AdamOptimizer(BaseOptimizer):
    def __init__(self, lrate:float = 0.001, b1:float = 0.9, b2:float = 0.999):
        self.lrate = lrate
        self.b1 = b1
        self.b2 = b2
        self.m = 0
        self.v = 0
        self.e = 0.001
        self.i = 1
    
    def dW(self, g:np.array):
        self.m = self.b1 * self.m + (1 - self.b1)*g
        self.v = self.b2 *self.v + (1 - self.b2)*(g**2)
        m_hat = self.m / (1 - np.power(self.b1, self.i))
        v_hat = self.v / (1 - np.power(self.b2, self.i))
        self.i += 1
        return -self.lrate * m_hat / (np.sqrt(v_hat) + self.e)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import NeuralNetwork_generic as NNg

    input = InputLayer(units=1)
    dl1 = DenseLayer(units=24)
    input.setFLink(dl1)
    dl1.setBLink(input)
    dl2 = DenseLayer(units=32)
    dl1.setFLink(dl2)
    dl2.setBLink(dl1)
    output = DenseLayer(units=2)
    dl2.setFLink(output)
    output.setBLink(dl2)
    dl1.setOptimizer(BaseOptimizer(lrate=0.031))
    dl2.setOptimizer(BaseOptimizer(lrate=0.031))
    output.setOptimizer(BaseOptimizer(lrate=0.031))

    dl1.compile()
    dl2.compile()
    output.compile()

    t,y = NNg.lissajous_curve()

    print(np.sum((input.forward(t)-y)**2)/len(y))

    for i in range(40000):
        res = input.forward(t)
        output.backward(2*(res-y))

    print(np.sum((input.forward(t)-y)**2)/len(y))

    t_n,y_n = NNg.lissajous_curve(N=150)
    y_pred = input.forward(t_n)
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
