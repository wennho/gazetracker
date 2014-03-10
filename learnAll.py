from imports import *
from extract_feature import extractFeatures
from sklearn.svm import SVR
from sklearn import cross_validation
import collect_calibrate
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from util import scale, scaleMatrix
from sklearn.decomposition import PCA



def getFeatures():
    dataX = []
    dataY = []
    for i in range(9):
        imgFile = 'calibrate_' + str(i) + '.png'
        image = cv2.imread(imgFile)
        featX, featY = extractFeatures(image, False)
        dataX.append(featX)
        dataY.append(featY)
        print 'Finished processing image', i

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    np.save('featureDataX', dataX)
    np.save('featureDataY', dataY)
    print 'saved data'


def crossValidate(dataX, dataY, eps, C, verbose):

    circleLoc = np.array(collect_calibrate.circleLoc)

    error = []
    trainError = []

    # using leave-one-out validation
    N = dataX.shape[0]
    for i in range(N):
        if verbose:
            print 'in iteration', i
        select = np.ones((N,), dtype=bool)
        select[i] = False

        testX = dataX[i]
        testY = dataY[i]
        testLoc = circleLoc[i]

        trainDataX = dataX[select]
        trainDataY = dataY[select]

        trainCircleLoc = circleLoc[select]

        # if usePCA:
        #     pca = PCA()
        #     pca.fit(trainData)
        #     trainData = pca.transform(trainData)
        #     test = pca.transform(test)

        xLabel = trainCircleLoc[:, 0]
        yLabel = trainCircleLoc[:, 1]

        # train linear SVMs
        y_svm = SVR(kernel='linear', C=C, epsilon=eps)
        x_svm = SVR(kernel='linear', C=C, epsilon=eps)

        x_class = x_svm.fit(trainDataX, xLabel)
        y_class = y_svm.fit(trainDataY, yLabel)

        xPredict = x_class.predict(trainDataX)
        yPredict = y_class.predict(trainDataY)

        predict = np.array(zip(xPredict, yPredict))
        distances = norm(predict - trainCircleLoc, axis=1)
        trainErr = np.mean(distances)
        trainError.append(trainErr)

        # print predict
        # print distances

        testPredict = (x_class.predict(testX), y_class.predict(testY))
        testPredict = np.array(testPredict)
        # print testPredict
        diff = testPredict.T - testLoc
        err = norm(diff)
        if verbose:
            print '\ttrain error:', trainErr
            print '\tdiff:', diff
            print '\ttest error:', err
        error.append(err)
        # sys.exit()

    meanError = np.mean(error)
    medianError = np.median(error)

    if verbose:
        print 'average train error:', np.mean(trainError)
        print 'average test error:', meanError
        print 'median test error:', medianError

    return trainError, meanError, medianError


if __name__ == "__main__":
    # getFeatures()

    useScale = False
    # usePCA = False

    dataX = np.load('featureDataX.npy')
    dataY = np.load('featureDataY.npy')

    # combine to provide higher dimensions
    dataX = np.hstack((dataX, dataY))
    dataY = dataX

    if useScale:
        dataX = scaleMatrix(dataX)
        dataY = scaleMatrix(dataY)

    eps = 1
    C = 1e1
    currErr = 1000
    paramChanged = True

    while paramChanged:

        paramChanged = False

        # test eps
        newEps = eps * 1.1
        trainError, meanError, medianError = crossValidate(dataX, dataY, newEps, C, False)
        if medianError < currErr:
            eps = newEps
            paramChanged = True
            currErr = medianError
        else:
            newEps = eps * 0.9
            trainError, meanError, medianError = crossValidate(dataX, dataY, newEps, C, False)
            if medianError < currErr:
                eps = newEps
                paramChanged = True
                currErr = medianError

        # test C
        newC = C * 1.1
        trainError, meanError, medianError = crossValidate(dataX, dataY, eps, newC, False)
        if medianError < currErr:
            C = newC
            paramChanged = True
            currErr = medianError
        else:
            newC = C * 0.9
            trainError, meanError, medianError = crossValidate(dataX, dataY, eps, newC, False)
            if medianError < currErr:
                C = newC
                paramChanged = True
                currErr = medianError

        print 'medianError:', medianError
        print 'meanError:', meanError
        print 'C:', C
        print 'eps:', eps