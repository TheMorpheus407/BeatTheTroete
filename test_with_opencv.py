import numpy as np
import cv2
from mss import mss
import numpy
import pytesseract

#bounds = {'top': 600+600, 'left': 3840*2+1000+650, 'width': 1840-1720, 'height': 1200-1150}
bounds = {'top': 600, 'left': 3840*2+1000, 'width': 1840, 'height': 1200}

screen = mss()
while True:
    screen_img = screen.grab(bounds)
    im = numpy.array(screen_img, dtype=numpy.uint8)
    im = numpy.flip(im[:, :, :3], 2)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (thresh, im) = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hist = cv2.calcHist([im], [0], None, [2], [0,256])
    diff = (hist[0] - hist[1])[0]
    #print(diff) #MISS == 74.04

    text = pytesseract.image_to_string(im)
    cv2.imshow('screen', np.array(im))
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
    print(text)
