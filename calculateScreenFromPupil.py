from imports import *


def calculateXprime(x, d, r, theta):
    x = float(x)
    d = float(d)
    r = float(r)
    theta = float(theta)

    # find P
    coeff = (x / (d - r)) ** 2
    a = 1 + coeff
    negB = 2 * d * coeff
    c = d ** 2 * coeff - r ** 2
    pz = (negB + math.sqrt(negB ** 2 - 4 * a * c)) / (2 * a)
    px2 = r ** 2 - pz ** 2
    px = math.sqrt(px2)

    # find p_norm
    lengthP = math.sqrt((r - pz) ** 2 + px2)

    p_theta = math.acos(0.5 * lengthP / r) # in radians
    p_norm = lengthP * math.sin(p_theta)

    xPrime = p_norm / math.cos(math.radians(90 - theta) - math.atan2(pz, px))
    return xPrime


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print 'Usage: python ' + __file__ + ' <x> <d> <r> <theta>'
        sys.exit()

    xMax = float(sys.argv[1])
    d = float(sys.argv[2])
    r = float(sys.argv[3])
    theta = float(sys.argv[4])

    xDomain = np.arange(0,xMax,0.1)
    y = np.zeros((3, xDomain.shape[0]))

    dRange = (d, d*2, d*4)


    for i, x in enumerate(xDomain):
        for j,d in enumerate(dRange):
            y[j,i] = calculateXprime(x, d, r, theta)

    plt.plot(xDomain, y[0], 'r', xDomain, y[1], 'g', xDomain, y[2], 'b')
    plt.show()