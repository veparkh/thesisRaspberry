import cv2
import numpy as np
def imageToByterray(img):
    img2 = cv2.resize(img,(300,700),1,1,cv2.INTER_AREA)
    cv2.imshow("changed", img2)
    cv2.waitKey(0)
    result, byteArray = cv2.imencode(".png", img)
    print("array size", len(byteArray))
    if(result):
         return byteArray
    else:
        return None




