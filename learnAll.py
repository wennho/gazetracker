from imports import *
from extract_feature import extractFeatures
from sklearn.svm import SVR
import collect_calibrate
from numpy.linalg import norm
from util import scale, scaleMatrix
from sklearn.decomposition import PCA
from eyeHough import *
import pickle

KERNEL = 'linear'


def getFeatures(writeImg):
    dataX = []
    dataY = []

    settings = {
        'face_cascade': cv2.CascadeClassifier("haarcascade_frontalface_default.xml"),
        'eye_cascade': cv2.CascadeClassifier('haarcascade_eye.xml'),
        'templateLeft': getLeftEyeTemplate(),
        'templateRight': getRightEyeTemplate(),
    }

    for i in range(50):
        imgFile = 'calibrate_' + str(i) + '.png'
        image = cv2.imread(imgFile)
        imageNum = i if writeImg else None
        featX, featY = extractFeatures(image, settings, imageNum)
        dataX.append(featX)
        dataY.append(featY)
        print 'Finished processing image', i

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    np.save('featureDataX', dataX)
    np.save('featureDataY', dataY)
    print 'saved data'


def crossValidate(dataX, dataY, labels, eps, C, verbose):
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
        testLoc = labels[i]

        trainDataX = dataX[select]
        trainDataY = dataY[select]

        trainCircleLoc = labels[select]

        # if usePCA:
        #     pca = PCA()
        #     pca.fit(trainData)
        #     trainData = pca.transform(trainData)
        #     test = pca.transform(test)

        xLabel = trainCircleLoc[:, 0]
        yLabel = trainCircleLoc[:, 1]

        # train linear SVMs
        y_svm = SVR(kernel=KERNEL, C=C, epsilon=eps)
        x_svm = SVR(kernel=KERNEL, C=C, epsilon=eps)

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
    trainErrorMean = np.mean(trainError)
    if verbose:
        print 'average train error:', trainErrorMean
        print 'average test error:', meanError
        print 'median test error:', medianError

    y_svm = SVR(kernel=KERNEL, C=C, epsilon=eps)
    x_svm = SVR(kernel=KERNEL, C=C, epsilon=eps)
    xSVM = x_svm.fit(dataX, labels[:, 0])
    ySVM = y_svm.fit(dataY, labels[:, 1])

    return {
        'trainErr': trainErrorMean,
        'trainErrorData': trainError,
        'errorData': error,
        'meanErr': meanError,
        'medianErr': medianError,
        'xSVM': xSVM,
        'ySVM': ySVM,
        'C': C,
        'eps': eps,
    }


def learn(dataX, dataY, labels):
    epsMin = 0.1
    epsMax = 10
    cMin = 1
    cMax = 1000

    #grid search for best hyper-param
    eps = epsMin
    bestResult = {'meanErr': 10000}
    while eps < epsMax:

        C = cMin
        while C < cMax:

            result = crossValidate(dataX, dataY, labels, eps, C, False)
            if result['meanErr'] < bestResult['meanErr']:
                bestResult = result
                print 'trainError:', result['trainErr']
                print 'medianError:', result['medianErr']
                print 'meanError:', result['meanErr']
                print 'C:', C
                print 'eps:', eps
            C *= 2
        eps *= 2

    return bestResult


if __name__ == "__main__":
    getFeatures(False)

    useScale = False
    # usePCA = False

    dataX = np.load('featureDataX.npy')
    dataY = np.load('featureDataY.npy')

    # combine to provide higher dimensions
    # dataX = np.hstack((dataX, dataY))
    # dataY = dataX

    if useScale:
        dataX = scaleMatrix(dataX)
        dataY = scaleMatrix(dataY)

    bestResult = learn(dataX, dataY, np.array(collect_calibrate.circleLoc))
    # print bestResult
    pickle.dump(bestResult, open('learnResult.pickle', 'wb'))
