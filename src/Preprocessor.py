import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class Preprocessor:
    def __init__(self):
        pass

    def getData(self):
        df = pd.read_csv('../data/train.csv')
        data = df.values.astype(np.float32)
        np.random.shuffle(data)

        X = data[:, 1:]
        Y = data[:, 0].astype(np.int32)

        Xtrain = X[:-1000]
        Ytrain = Y[:-1000]
        Xtest  = X[-1000:]
        Ytest  = Y[-1000:]

        # normalize the data
        mu = Xtrain.mean(axis=0)
        std = Xtrain.std(axis=0)
        np.place(std, std == 0, 1)
        Xtrain = (Xtrain - mu) / std
        Xtest = (Xtest - mu) / std
        return Xtrain, Xtest, Ytrain, Ytest

if __name__=="__main__":
    ins = Preprocessor()
    Xtrain, Xtest, Ytrain, Ytest = ins.getData()
