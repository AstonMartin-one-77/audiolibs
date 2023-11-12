
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
        return -self.lrate*g
    
    def db(self, g:np.array):
        return -self.lrate*np.sum(g, axis=1, keepdims=True)

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
        dAdZ = self.dG(d)
        dZdA = dAdZ.T @ self.W
        self.W += self.optimizer.dW(dAdZ @ self.saveAn.T) / dAdZ.size
        self.b += self.optimizer.db(dAdZ) / dAdZ.size
        self.bLayer.backward(dZdA.T)

    def forward(self, aVal:np.array):
        self.saveAn = aVal
        output = self.G(self.W @ aVal + self.b)
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
