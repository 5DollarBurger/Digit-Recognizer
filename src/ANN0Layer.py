import numpy as np
from sklearn.utils import shuffle
from CoreClasses.ANNBaseClass import ANNBaseClass
import os
import json

PWD = os.path.abspath(os.path.dirname(__file__))


class ANN0Layer(ANNBaseClass):
    def __init__(self):
        self.confDict = json.load(fp=open(f"{PWD}/conf.json", "r"))

    def _getCost(self, y, t):
        """
        Evaluate loss for a given set of predictions
        """
        loss = (-t * np.log(y)).sum()
        return loss

    def _getW2Grad(self, X, y, t):
        """
        Calculate loss gradient w.r.t. W2
        """
        return X.T.dot(t - y)

    def _getb2Grad(self, y, t):
        """
        Calculate loss gradient w.r.t. b2
        """
        return (t - y).sum(axis=0)

    def getClassProbs(self, X):
        """
        Calculate probability of classes using softmax
        """
        A = X.dot(self.W2) + self.b2
        num = np.exp(A)
        den = num.sum(axis=1, keepdims=True)
        return num / den

    def predict(self, X):
        """
        Predict class using argmax
        """
        py = self.getClassProbs(X=X)
        return np.argmax(py, axis=1)

    def train(self, Xtrain, Ytrain, Xtest, Ytest, B=None,
              learnRate=0.00004, reg=0.01, iter=30,
              mu=0.9):
        self.N, self.D = Xtrain.shape
        self.K = len(np.unique(Ytrain))
        if B is None:
            B = self.N
        numBatch = self._getNumBatch(N=self.N, B=B)

        # convert Ytrain and Ytest to (N x K) matrices of indicator variables
        YtrainEnc = self._getEncodedY(Y=Ytrain, K=self.K) # describe as probabilities
        YtestEnc = self._getEncodedY(Y=Ytest, K=self.K)

        # initialize params
        self.W2 = np.random.randn(self.D, self.K) / np.sqrt(self.D)
        self.b2 = np.zeros(self.K)

        # calculate current cost
        LLtrain = []
        LLtest = []
        CRtest = []
        for i in range(iter):
            tmpX, tmpT = shuffle(Xtrain, YtrainEnc)
            for j in range(numBatch):
                x = tmpX[j * B:(j + 1) * B, :]
                t = tmpT[j * B:(j + 1) * B, :]
                currB = len(x)

                py = self.getClassProbs(X=x)
                # LLtrain.append(self._getCost(y=py, t=YtrainEnc))

                # run gradient descent
                gradW2 = self._getW2Grad(X=x, y=py, t=t)
                gradb2 = self._getb2Grad(y=py, t=t)
                remainCorr = currB / B
                self.W2 += remainCorr * learnRate * gradW2 - reg * self.W2
                self.b2 += remainCorr * learnRate * gradb2 - reg * self.b2

            pyTest = self.getClassProbs(X=Xtest)
            LLtest.append(self._getCost(y=pyTest, t=YtestEnc))

            if i % 5 == 0:
                print("Cost at iteration %d: %.6f" % (i, LLtest[-1]))

        return LLtrain, LLtest


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.Preprocessor import Preprocessor

    np.random.seed(123)
    iter = 30
    insPrep = Preprocessor()
    Xtrain, Xtest, Ytrain, Ytest = insPrep.getData()
    ins = ANN0Layer()
    LLtrain, LLtest = ins.train(Xtrain=Xtrain, Ytrain=Ytrain,
                                Xtest=Xtest, Ytest=Ytest,
                                B=None, learnRate=0.00004, reg=0.01, iter=20)
    iterList = range(len(LLtest))
    plt.plot(#iterList, np.divide(LLtrain, len(Ytrain)),
             iterList, np.divide(LLtest, len(Ytest)))
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    # plt.legend(["Training set", "Test set"])
    plt.show()

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    Ypred = ins.predict(X=Xtest)
    cm = confusion_matrix(Ytest, Ypred)

    ax = sns.heatmap(cm, annot=True, cmap='YlGnBu', fmt="d")
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    plt.show()
