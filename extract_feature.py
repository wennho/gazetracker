from imports import *
from detectEyeShape import getEyeFeatures


def extractFeatures(image, imageNum, verbose):
    featX = []  # bias term
    featY = []
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # only process one face
    if len(faces) == 0:
        sys.exit()

    (x, y, w, h) = faces[0]

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    indices = eyes[:, 1].argsort()[:2]
    eyes = eyes[indices]  # take top 2 occurrences only
    eyes = eyes[eyes[:, 0].argsort()]  # sort by x-axis

    eyeNum = 0

    for (ex, ey, ew, eh) in eyes:

        eyeNum += 1
        if eyeNum == 2:
            ew += 15

        # if verbose and len(sys.argv) > 2:
        # eyeColorImg = roi_color[ey:ey + eh, ex:ex + ew]
        # cv2.imwrite('testeye' + str(eyeNum) + '_' + str(imageNum) + '.png', eyeColorImg)

        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        eyeGrayImg = roi_gray[ey:ey + eh, ex:ex + ew]
        eyeFeat = getEyeFeatures(eyeGrayImg, False)

        featX.append(eyeFeat['bottom'][0] + x + ex)
        featY.append(eyeFeat['bottom'][1] + y + ey)

        featX.append(eyeFeat['pupil'][0])
        featY.append(eyeFeat['pupil'][1])

    roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

    if verbose:
        plt.imshow(roi_color)
        plt.show()

    return featX, featY


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print 'Usage: python ' + __file__ + ' <image> <imageNum>'
        sys.exit()

    imgFile = sys.argv[1]
    image = cv2.imread(imgFile)
    featX, featY = extractFeatures(image, int(sys.argv[2]), True)
    print 'x features:', featX
    print 'y features:', featY