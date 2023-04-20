
import numpy as np
import fft_Q32 as ft

class AEC_Q:
    def __init__(self, blockSz, filterSz, stepSzQ15, lossQ15, initPwrQ15):
        self.blockSz = int(blockSz)
        self.filterSz = int(filterSz)
        self.stepSz = int(stepSzQ15)
        self.normCff = int(1 << 15) - int(lossQ15)
        self.loss = int(lossQ15)
        self.inBuf = np.ndarray(self.blockSz+self.filterSz, dtype='int16')
        self.inBuf.fill(0)
        self.wRe = np.ndarray(len(self.inBuf), dtype='int64')
        self.wRe.fill(0)
        self.wIm = np.ndarray(len(self.inBuf), dtype='int64')
        self.wIm.fill(0)
        self.normIn = np.ndarray(len(self.inBuf), dtype='int64')
        self.normIn.fill(int(initPwrQ15))
        self.tmpRe = np.ndarray(len(self.inBuf), dtype='int64')
        self.tpmIm = np.ndarray(len(self.inBuf), dtype='int64') 
        self.inFft = ft.FFT_Q32(len(self.inBuf))
        self.ifft = ft.FFT_Q32(len(self.inBuf))
        self.wFft = ft.FFT_Q32(len(self.inBuf))
    
    def echoCancel(self, micQ15, spQ15):
        cycles = len(micQ15) // self.blockSz

        for cycle in range(cycles):
            startIdx = cycle*self.blockSz
            endIdx = startIdx+self.blockSz
            self.inBuf[self.filterSz:] = spQ15[startIdx:endIdx]
            inRe, inIm = self.inFft.fft32(self.inBuf)

            self.tmpRe = self.wRe*inRe - self.wIm*inIm
            self.tmpRe >>= 15
            self.tpmIm = self.wRe*inIm + self.wIm*inRe
            self.tpmIm >>= 15
            
            echoRe, echoIm = self.ifft.ifft32(self.tmpRe, self.tpmIm)

            echoRe[self.filterSz:] >>= 15
            micQ15[startIdx:endIdx] -= echoRe[self.filterSz:]
            self.tmpRe[self.filterSz:] = micQ15[startIdx:endIdx]*self.stepSz
            self.tmpRe[:self.filterSz] = 0

            stepRe, stepIm = self.wFft.fft32(self.tmpRe)

            self.tmpRe = inRe*inRe
            self.tmpRe += inIm*inIm
            self.tmpRe += int(1 << 14)
            self.tmpRe >>= 15
            self.tmpRe *= self.normCff
            self.tmpRe += self.normIn*self.loss
            self.tmpRe += int(1 << 14)
            self.normIn = self.tmpRe >> 15

            self.tmpRe = stepRe*inRe + stepIm*inIm
            self.tmpRe //= self.normIn
            self.wRe += self.tmpRe
            self.tmpIm = stepIm*inRe - stepRe*inIm
            self.tmpIm //= self.normIn
            self.wIm += self.tmpIm

            self.inBuf[0:self.filterSz] = self.inBuf[self.blockSz:self.blockSz+self.filterSz]
        
        return micQ15


class AEC:
    def __init__(self, blockSz, filterSz, stepSz, initPwr, loss):
        self.blockSz = blockSz
        self.filterSz = filterSz
        self.stepSz = stepSz
        self.loss = loss
        self.normCff = 1 - loss
        self.inBuf = np.ndarray(blockSz+filterSz, dtype="float")
        self.inBuf.fill(0)
        self.tmpBuf = np.ndarray(len(self.inBuf), dtype="float")
        self.tmpBuf.fill(0)
        self.wBuf = np.ndarray(len(self.inBuf), dtype="complex")
        self.wBuf.fill(0)
        self.normSp = np.ndarray(len(self.inBuf), dtype="float")
        self.normSp.fill(initPwr)

    
    def echoCancel(self, mic, sp):
        cycles = len(mic) // self.blockSz

        for cycle in range(cycles):
            startIdx = cycle*self.blockSz
            endIdx = startIdx+self.blockSz
            self.inBuf[self.filterSz:] = sp[startIdx:endIdx]
            spCmplx = np.fft.fft(self.inBuf)
            dsCmplx = np.fft.ifft(spCmplx*self.wBuf)
            mic[startIdx:endIdx] -= dsCmplx.real[self.filterSz:]
            
            self.tmpBuf[self.filterSz:] = mic[startIdx:endIdx]*self.stepSz
            corrCmplx = np.fft.fft(self.tmpBuf)
            corrCmplx *= np.conj(spCmplx)

            spCmplx *= np.conj(spCmplx)
            self.normSp *= self.loss
            self.normSp += spCmplx.real*self.normCff

            self.wBuf += corrCmplx / self.normSp


        return mic
