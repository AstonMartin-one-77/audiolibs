
import numpy as np

class ComplexBuf:
    def __init__(self, size):
        self.re = np.ndarray(shape=(size), dtype='int32')
        self.re.fill(0)
        self.im = np.ndarray(shape=(size), dtype='int32')
        self.im.fill(0)

class AEC:
    def __init__(self, blockSz, filterSz):
        self.blockSz = blockSz
        self.filterSz = filterSz
        self.inputBuf = ComplexBuf(blockSz+filterSz)

objMyClass = AEC(10,10)
print(objMyClass.inputBuf.re, objMyClass.inputBuf.im)
