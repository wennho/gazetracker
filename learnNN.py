from imports import *
from sklearn.svm import SVR
from collect_calibrate import circleLoc
from numpy.linalg import norm
from util import scale, scaleMatrix
from sklearn.decomposition import PCA
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

if __name__ == "__main__":

    useScale = True
    usePCA = True

    data = np.load('featureData.npy')

    if useScale:
        data = scaleMatrix(data)

    circleLoc, scaleFactor = scale(np.array(circleLoc))
    data[:,:2] = circleLoc

    error = []

    # using leave-one-out validation
    m = data.shape[0]

    for i in range(m):
        print 'in iteration', i
        select = np.ones((m,), dtype=bool)
        select[i] = False

        test = data[i]
        testLoc = circleLoc[i]

        trainData = data[select]
        trainCircleLoc = circleLoc[select]
        scale = np.max(trainCircleLoc)
        trainCircleLoc = trainCircleLoc / scale

        if usePCA:
            pca = PCA()
            pca.fit(trainData)
            trainData = pca.transform(trainData)
            test = pca.transform(test)[0]


        # train NN
        n = trainData.shape[1]
        net = buildNetwork(n, n * 2, n, 2, bias=True, hiddenclass=TanhLayer)
        ds = SupervisedDataSet(n, 2)
        ds.setField('input', trainData)
        ds.setField('target', trainCircleLoc)
        trainer = BackpropTrainer(net, ds)
        trainer.trainUntilConvergence()
        predict = np.zeros(trainCircleLoc.shape)
        for j in range(m - 1):
            predict[j] = net.activate(trainData[j])

        distances = norm(predict - trainCircleLoc, axis=1) * scaleFactor

        print '\ttrain error:', np.mean(distances)

        print predict
        print distances

        testPredict = net.activate(test)
        print testPredict
        diff = testPredict.T - testLoc
        err = norm(diff) * scaleFactor
        print '\ttest error:', err
        error.append(err)

    print 'average test error:', np.mean(error)
    print 'median test error:', np.median(error)

