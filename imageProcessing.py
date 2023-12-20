import cv2

def imageToBMPArray(img):
    img2 = cv2.resize(img,(320,700),1,1,cv2.INTER_AREA)
    cv2.imshow("changed", img2)
    result, byteArray = cv2.imencode(".bmp", img2)
    if(result):
         return byteArray
    else:
        return None




