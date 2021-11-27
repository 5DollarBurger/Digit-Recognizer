import numpy as np


class ANNBaseClass:
    def __init__(self):
        pass

    def _getEncodedY(self, Y, K):
        """
        Onehotencodes class series to represent class probabilities
        """
        N = len(Y)
        YEnc = np.zeros(shape=(N, K))
        for i in range(N):
            YEnc[i, Y[i]] = 1
        return YEnc.astype(np.float32)

    def _getNumBatch(self, N, B):
        """
        Calculate number of batches based on batch size
        """
        return int(np.ceil(N / B))

    def _getRelu(self, a):
        return a * (a > 0)