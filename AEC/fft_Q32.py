
import numpy as np

maxFftLine = 4096
cosShift = maxFftLine >> 2
sinCosLen = maxFftLine + cosShift
maxVal_Q32 = np.iinfo(np.dtype('int32')).max
twiddle_sinCos_Q32 = np.ndarray(maxFftLine, 
                                buffer=np.array([int(np.round(maxVal_Q32*np.sin(2*np.pi*x/maxFftLine)))
                                                 for x in range(sinCosLen)]), dtype='int32')

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
