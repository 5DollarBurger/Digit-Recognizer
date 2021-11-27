import numpy as np
import theano
import theano.tensor as T
from CoreClasses.ANNBaseClass import ANNBaseClass


class ANNTheano(ANNBaseClass):
    def __init__(self, M):
        self.M = M

    def setTensors(self, Xtrain, Ytrain, learnRate=0.0004, reg=0.01):
        """
        Set variable relations
        """
        self.N, self.D = Xtrain.shape
        self.K = len(np.unique(Ytrain))

        # initialize params
        W1_init = np.random.randn(self.D, self.M) / np.sqrt(self.D)
        b1_init = np.zeros(self.M)
        W2_init = np.random.randn(self.M, self.K) / np.sqrt(self.M)
        b2_init = np.zeros(self.K)

        # define theano variables
        thX = T.matrix('X')  # tensor for indep var
        thT = T.matrix('T')  # tensor for target
        W1 = theano.shared(W1_init, 'W1')
        b1 = theano.shared(b1_init, 'b1')
        W2 = theano.shared(W2_init, 'W2')
        b2 = theano.shared(b2_init, 'b2')

        # define activation functions
        thZ = self._getRelu(a=thX.dot(W1) + b1)
        thpy = T.nnet.softmax(c=thZ.dot(W2) + b2)

        # define cost function
        cost = -(thT * T.log(thpy)).sum() \
               + reg * ((W1 * W1).sum()
                      + (b1 * b1).sum()
                      + (W2 * W2).sum()
                      + (b2 * b2).sum())

        # define updates
        update_W1 = W1 - learnRate * T.grad(cost, W1)
        update_b1 = b1 - learnRate * T.grad(cost, b1)
        update_W2 = W2 - learnRate * T.grad(cost, W2)
        update_b2 = b2 - learnRate * T.grad(cost, b2)

        # define train operation
        self.trainFunc = theano.function(
            inputs=[thX, thT], outputs=cost,
            updates=[(W1, update_W1), (b1, update_b1),
                     (W2, update_W2), (b2, update_b2)],
        )

        # define prediction operation
        thY = T.argmax(thpy, axis=1)
        self.predFunc = theano.function(inputs=[thX], outputs=thY)

    def train(self, Xtrain, Ytrain, iter=30, B=None):
        """
        Perform batch gradient decent to optimize params in predict function
        """
        if B is None:
            B = self.N
        numBatch = self._getNumBatch(N=self.N, B=B)

        # convert Ytrain and Ytest to (N x K) matrices of indicator variables
        YtrainEnc = self._getEncodedY(Y=Ytrain, K=self.K) # describe as probabilities
        YtestEnc = self._getEncodedY(Y=Ytest, K=self.K)

        LLtrain = []
        for i in range(iter):
            tmpX, tmpT = Xtrain, YtrainEnc  #TODO: shuffle?
            for j in range(numBatch):
                x = tmpX[j * B:(j + 1) * B, :]
                t = tmpT[j * B:(j + 1) * B, :]
                cost = self.trainFunc(x, t)
                LLtrain.append(cost)

            if i % 5 == 0:
                print("Cost at iteration %d: %.6f" % (i, LLtrain[-1]))

        return LLtrain

    def predict(self, X):
        """
        Predict class using argmax
        """
        Y = self.predFunc(X)
        return Y


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from src.Preprocessor import Preprocessor

    np.random.seed(123)
    iter = 30
    insPrep = Preprocessor()
    Xtrain, Xtest, Ytrain, Ytest = insPrep.getData()
    ins = ANNTheano(M=300)
    ins.setTensors(Xtrain=Xtrain, Ytrain=Ytrain, learnRate=0.0004, reg=0.01)
    LLtrain = ins.train(Xtrain=Xtrain, Ytrain=Ytrain, iter=20, B=500)

    plt.plot(np.divide(LLtrain, len(Ytrain)))
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