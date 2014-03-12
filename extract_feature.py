from imports import *
from detectEyeShape import getEyeFeatures


def getEyeFacePos(image, face_cascade, eye_cascade, imageSaveName=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]

    result = {'face': (x, y, w, h)}

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_color = image[y:y + h, x:x + w]
    roi_gray = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    indices = eyes[:, 1].argsort()[:2]
    eyes = eyes[indices]  # take top 2 occurrences only
    eyes = eyes[eyes[:, 0].argsort()]  # sort by x-axis

    eyeNum = 0

    for (ex, ey, ew, eh) in eyes:

        eyeNum += 1
        if eyeNum == 2:
            ew += 15

        if imageSaveName:
            eyeColorImg = roi_color[ey:ey + eh, ex:ex + ew]
            cv2.imwrite(imageSaveName + '.png', eyeColorImg)

        # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        eyeName = 'eyeLeft' if eyeNum == 1 else 'eyeRight'
        result[eyeName] = ((ex + x, int(ey + y + eh * 0.15)), (ex + x + ew, int(ey + y + eh * .85)))

    return result


def extractFeatures(image, face_cascade, eye_cascade, verbose, imageSaveName=None):
    featX = []
    featY = []


    # eyeGrayImg = roi_gray[ey:ey + eh, ex:ex + ew]
    # eyeFeat = getEyeFeatures(eyeGrayImg, False)

    # featX.append(eyeFeat['bottom'][0] + x + ex)
    # featY.append(eyeFeat['bottom'][1] + y + ey)

    # featX.append(eyeFeat['pupil'][0])
    # featY.append(eyeFeat['pupil'][1])


    if verbose:
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        plt.imshow(roi_color)
        plt.show()

    return featX, featY


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print 'Usage: python ' + __file__ + ' <image> <imageNum>'
        sys.exit()

    imgFile = sys.argv[1]
    image = cv2.imread(imgFile)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    featX, featY = extractFeatures(image, sys.argv[2], True, True)
    print 'x features:', featX
    print 'y features:', featY