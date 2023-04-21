
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

        if cycles == 0:
            return

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

class RingBuffer:
    def __init__(self, bufSz):
        self.buf = np.ndarray(bufSz, dtype="float")
        self.wIdx = 0
        self.rIdx = 0
    
    def set(self, frame):

        if self.wIdx >= self.rIdx:
            endSpace = len(self.buf)-self.wIdx
            free = endSpace+self.rIdx-1
            if endSpace > len(frame):
                self.buf[self.wIdx:self.wIdx+len(frame)] = frame
                self.wIdx += len(frame)
            elif free >= len(frame):
                self.buf[self.wIdx:] = frame[:endSpace]
                self.buf[:len(frame)-endSpace] = frame[endSpace:]
                self.wIdx = len(frame)-endSpace
            else:
                # TODO: exception
                pass
        else:
            free = self.rIdx-self.wIdx-1
            if free >= len(frame):
                self.buf[self.wIdx:self.wIdx+len(frame)] = frame
                self.wIdx += len(frame)
            else:
                # TODO: exception
                pass
    
    def getPlenum(self):
        if self.wIdx >= self.rIdx:
            return self.wIdx-self.rIdx
        else:
            return len(self.buf)-self.rIdx+self.wIdx
    

    def get(self, size):
        if self.wIdx >= self.rIdx:
            busy = self.wIdx-self.rIdx
            if busy < size:
                return
            else:
                res = np.copy(self.buf[self.rIdx:self.rIdx+size])
                self.rIdx += size
                return res
        else:
            endBusy = len(self.buf)-self.rIdx
            busy = endBusy+self.wIdx
            if endBusy > size:
                res = np.copy(self.buf[self.rIdx:self.rIdx+size])
                self.rIdx += size
                return res
            elif busy >= size:
                res = np.ndarray(size, dtype="float")
                res[:endBusy] = self.buf[self.rIdx:]
                res[endBusy:] = self.buf[:size-endBusy]
                self.rIdx = size-endBusy
                return res


class AEC_Align:
    def __init__(self):
        self.spRingBuf = RingBuffer(48000) # 1 sec delay if it has 48kHz
        self.micRingBuf = RingBuffer(48000) # 1 sec delay if it has 48kHz
        self.blockSz = 4096
        self.filterSz = 4096
        self.alignStepSz = 128
        self.aec = AEC(self.blockSz, self.filterSz, 0.032, 0.01, 0.98)
        self.micBuf = np.ndarray(self.filterSz+self.alignStepSz, dtype="float")
        self.isAlign = False
        self.cnvVal = 0

    def setMic(self, mic):
        if self.isAlign == False:
            cycles = len(mic) // self.alignStepSz
            for cycle in range(cycles):
                start = cycle*self.alignStepSz
                end = start+self.alignStepSz
                self.micBuf[self.filterSz:] = mic[start:end]
                cnvRes = np.convolve(self.micBuf[self.alignStepSz:], self.spRingBuf.buf[:self.filterSz],"valid")
                crrCnv = cnvRes[0] / self.filterSz
                if crrCnv > 0.2:
                    if crrCnv >= self.cnvVal:
                        self.cnvVal = crrCnv
                        self.micBuf[:self.filterSz] = self.micBuf[self.alignStepSz:]
                    else:
                        self.micRingBuf.set(self.micBuf)
                        self.micRingBuf.set(mic[end:])
                        self.isAlign = True
                        break
                else:
                    self.micBuf[:self.filterSz] = self.micBuf[self.alignStepSz:]
            
            remain = len(mic)-cycles*self.alignStepSz
            if self.isAlign == False:
                self.micBuf[self.filterSz:self.filterSz+remain] = mic[len(mic)-remain:]
                cnvRes = np.convolve(self.micBuf[remain:remain+self.filterSz], self.spRingBuf.buf[:self.filterSz],"valid")
                crrCnv = cnvRes[0] / self.filterSz
                if crrCnv > 0.2:
                    if crrCnv >= self.cnvVal:
                        self.cnvVal = crrCnv
                        self.micBuf[:self.filterSz] = self.micBuf[remain:remain+self.filterSz]
                    else:
                        self.micRingBuf.set(self.micBuf)
                        self.micRingBuf.set(mic[end:])
                        self.isAlign = True
                else:
                    self.micBuf[:self.filterSz] = self.micBuf[remain:remain+self.filterSz]

    def setSp(self, sp):
        self.spRingBuf.set(sp)

    def echoCancel(self):
        if self.isAlign == True:
            sp = self.spRingBuf.get(self.blockSz)
            mic = self.micRingBuf.get(self.blockSz)
            if sp != None and mic != None:
                self.aec.echoCancel(mic, sp)
