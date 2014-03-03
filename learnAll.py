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
    data = []
    for i in range(9):
        imgFile = 'calibrate_' + str(i) + '.png'
        image = cv2.imread(imgFile)
        data.append(extractFeatures(image, False))
        print 'Finished processing image', i

    data = np.array(data)
    np.save('featureData', data)
    print 'saved data'


if __name__ == "__main__":
    # getFeatures()

    useScale = False
    # usePCA = False

    data = np.load('featureData.npy')

    if useScale:
        data = scaleMatrix(data)

    circleLoc = np.array(circleLoc)

    error = []

    # using leave-one-out validation
    N = data.shape[0]
    for i in range(N):
        print 'in iteration', i
        select = np.ones((N,), dtype=bool)
        select[i] = False

        test = data[i]
        testLoc = circleLoc[i]

        trainData = data[select]
        trainCircleLoc = circleLoc[select]

        # if usePCA:
        #     pca = PCA()
        #     pca.fit(trainData)
        #     trainData = pca.transform(trainData)
        #     test = pca.transform(test)

        xLabel = trainCircleLoc[:, 0]
        yLabel = trainCircleLoc[:, 1]

        # train linear SVMs
        eps = 50 if not useScale else 0.05
        y_svm = SVR(kernel='linear', C=1e3, epsilon=eps)
        x_svm = SVR(kernel='linear', C=1e3, epsilon=eps)

        x_class = x_svm.fit(trainData, xLabel)
        y_class = y_svm.fit(trainData, yLabel)

        xPredict = x_class.predict(trainData)
        yPredict = y_class.predict(trainData)

        predict = np.array(zip(xPredict, yPredict))
        distances = norm(predict - trainCircleLoc, axis=1)
        print '\ttrain error:', np.mean(distances)

        # print predict
        # print distances

        testPredict = (x_class.predict(test), y_class.predict(test))
        testPredict = np.array(testPredict)
        # print testPredict
        diff = testPredict.T - testLoc
        err = norm(diff)
        print '\ttest error:', err
        error.append(err)
        # sys.exit()

    print 'average test error:', np.mean(error)
    print 'median test error:', np.median(error)

