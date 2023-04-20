
import numpy as np

maxFftLine = 4096
cosShift = maxFftLine >> 2
sinCosLen = maxFftLine + cosShift
maxVal_Q32 = np.iinfo(np.dtype('int32')).max
twiddle_sinCos_Q32 = np.ndarray(maxFftLine, 
                                buffer=np.array([int(np.round(maxVal_Q32*np.sin(2*np.pi*x/maxFftLine)))
                                                 for x in range(sinCosLen)]), dtype='int32')

class FFT_Q32:
    def __init__(self, size):
        self.size = 2
        self.pwr2 = 1
        while self.size < size:
            self.size *= 2
            self.pwr2 += 1
        self.cosShift = self.size >> 2
        self.re = np.ndarray(self.size, dtype='int64')
        self.im = np.ndarray(self.size, dtype='int64')
        self.sinCos = np.ndarray(self.size+self.cosShift, 
                                 buffer=np.array([int(np.round(1048576*np.sin(2*np.pi*x/self.size))) 
                                                  for x in range(self.size+self.cosShift)]), dtype='int64')
        self.bitRev = np.ndarray(self.size, dtype='int16')
        self.bitRev.fill(0)
        for i in range(1,self.size):
            self.bitRev[i] = (self.bitRev[i >> 1] >> 1) | ((i & 1) << (self.pwr2 - 1))

    def fft32(self, re):
        for idx in range(len(re)):
            self.re[self.bitRev[idx]] = re[idx]
            self.im[self.bitRev[idx]] = 0
        for idx in range(len(re), self.size):
            self.re[self.bitRev[idx]] = 0
            self.im[self.bitRev[idx]] = 0
        
        for depth in range(1, self.pwr2+1):
            step = 1 << depth
            hStep = step >> 1
            sinStep = self.size >> depth

            for idxStep in range(0, self.size, step):
                for idx in range(hStep):
                    sinIdx = sinStep*idx
                    cosIdx = sinIdx+self.cosShift
                    top = idxStep+idx
                    bottom = top+hStep
                    accum = self.sinCos[cosIdx]*self.re[bottom]
                    accum -= self.sinCos[sinIdx]*self.im[bottom]
                    accum >>= 31
                    self.re[bottom] = self.re[top] - accum
                    self.re[top] += accum
                    accum = self.sinCos[cosIdx]*self.im[bottom]
                    accum += self.sinCos[sinIdx]*self.re[bottom]
                    accum >>= 31
                    self.im[bottom] = self.im[top] - accum
                    self.re[top] += accum

        return self.re, self.im 

    def ifft32(self, re, im):
        for idx in range(len(re)):
            self.re[self.bitRev[idx]] = re[idx]
            self.im[self.bitRev[idx]] = -im[idx]
        for idx in range(len(re), self.size):
            self.re[self.bitRev[idx]] = 0
            self.im[self.bitRev[idx]] = 0
        
        for depth in range(1, self.pwr2+1):
            step = 1 << depth
            hStep = step >> 1
            sinStep = self.size >> depth

            for idxStep in range(0, self.size, step):
                for idx in range(hStep):
                    sinIdx = sinStep*idx
                    cosIdx = sinIdx+self.cosShift
                    top = idxStep+idx
                    bottom = top+hStep
                    accum = self.sinCos[cosIdx]*self.re[bottom]
                    accum -= self.sinCos[sinIdx]*self.im[bottom]
                    accum >>= 31
                    self.re[bottom] = self.re[top] - accum
                    self.re[top] += accum
                    accum = self.sinCos[cosIdx]*self.im[bottom]
                    accum += self.sinCos[sinIdx]*self.re[bottom]
                    accum >>= 31
                    self.im[bottom] = self.im[top] - accum
                    self.re[top] += accum

        return self.re, self.im

def fft_Q32(re, bitRev, pwr2):
    outSz = len(bitRev)
    outRe = np.ndarray(shape=(outSz), dtype='int32')
    outIm = np.ndarray(shape=(outSz), dtype='int32')

    for idx in range(outSz):
        outRe[bitRev[idx]] = re[idx]
        outIm[bitRev[idx]] = 0
    
    for depth in range(1,pwr2+1):
        step = 1 << depth
        hStep = step >> 1
        sinCosStep = maxFftLine >> depth

        for idxStep in range(0,outSz,step):
            for count in range(hStep):
                sinIdx = sinCosStep*count
                cosIdx = sinIdx+cosShift
                top = idxStep+count
                bottom = top+hStep
                accum = np.int64(twiddle_sinCos_Q32[cosIdx])*np.int64(outRe[bottom])
                accum -= np.int64(twiddle_sinCos_Q32[sinIdx])*np.int64(outIm[bottom])
                accRe32 = accum >> 31
                accum = np.int64(twiddle_sinCos_Q32[cosIdx])*np.int64(outIm[bottom])
                accum += np.int64(twiddle_sinCos_Q32[sinIdx])*np.int64(outRe[bottom])
                accIm32 = accum >> 31
                outRe[bottom] = outRe[top] - accRe32
                outIm[bottom] = outIm[top] - accIm32
                outRe[top] += accRe32
                outIm[top] += accIm32
    return outRe, outIm

def ifft_Q32(re, im, bitRev, pwr2):
    outSz = len(bitRev)
    outRe = np.ndarray(shape=(outSz), dtype='int32')
    outIm = np.ndarray(shape=(outSz), dtype='int32')

    for idx in range(outSz):
        outRe[bitRev[idx]] = re[idx]
        outIm[bitRev[idx]] = -im[idx]
    
    for depth in range(1,pwr2+1):
        step = 1 << depth
        hStep = step >> 1
        sinCosStep = maxFftLine >> depth
        
        for idxStep in range(0,outSz,step):
            for count in range(hStep):
                sinIdx = sinCosStep*count
                cosIdx = sinIdx+cosShift
                top = idxStep+count
                bottom = top+hStep
                accum = np.int64(twiddle_sinCos_Q32[cosIdx])*np.int64(outRe[bottom])
                accum -= np.int64(twiddle_sinCos_Q32[sinIdx])*np.int64(outIm[bottom])
                accRe32 = accum >> 31
                accum = np.int64(twiddle_sinCos_Q32[cosIdx])*np.int64(outIm[bottom])
                accum += np.int64(twiddle_sinCos_Q32[sinIdx])*np.int64(outRe[bottom])
                accIm32 = accum >> 31
                outRe[bottom] = outRe[top] - accRe32
                outIm[bottom] = outIm[top] - accIm32
                outRe[top] += accRe32
                outIm[top] += accIm32
    return outRe, outIm
