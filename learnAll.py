from imports import *
from extract_feature import extractFeatures
from sklearn.svm import SVR
from sklearn import cross_validation
from collect_calibrate import circleLoc
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

    circleLoc = np.array(circleLoc)

    error = []
    trainError = []

    # using leave-one-out validation
    N = dataX.shape[0]
    for i in range(N):
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
        eps = 1 if not useScale else 0.05  # impt to tweak epsilon as features change
        y_svm = SVR(kernel='linear', C=1e3, epsilon=eps)
        x_svm = SVR(kernel='linear', C=1e3, epsilon=eps)

        x_class = x_svm.fit(trainDataX, xLabel)
        y_class = y_svm.fit(trainDataY, yLabel)

        xPredict = x_class.predict(trainDataX)
        yPredict = y_class.predict(trainDataY)

        predict = np.array(zip(xPredict, yPredict))
        distances = norm(predict - trainCircleLoc, axis=1)
        trainErr = np.mean(distances)
        trainError.append(trainErr)
        print '\ttrain error:', trainErr

        # print predict
        # print distances

        testPredict = (x_class.predict(testX), y_class.predict(testY))
        testPredict = np.array(testPredict)
        # print testPredict
        diff = testPredict.T - testLoc
        err = norm(diff)
        print '\ttest error:', err
        error.append(err)
        # sys.exit()

    print 'average train error:', np.mean(trainError)
    print 'average test error:', np.mean(error)
    print 'median test error:', np.median(error)

