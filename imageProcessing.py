import cv2
import cv2 as cv
import numpy as np


def highlight_color(img):
    original = img.copy()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([90, 40, 140])
    upper = np.array([140, 255, 255])
    mask = cv.inRange(hsv, lower, upper)
    # mask = cv2.dilate(mask,(3,3),anchor = (-1,-1),iterations=11,borderType=cv2.BORDER_DEFAULT)
    result = cv.bitwise_and(original, original, mask=mask)
    cv2.imshow("color transformation", result)
    return result


def find_lines(img):
    img_lines = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))

    lines = cv2.HoughLinesP(img, 1, 3.14 / 180, 100, minLineLength=50,maxLineGap=10)

    if lines is not None:
        print(len(lines))
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_4)
    cv2.imshow("lines", img_lines)


def find_contours(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("ff", img)
    contours, tree = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    imgContour = np.zeros_like(img)
    img_contour_text = np.zeros_like(img)
    # cv2.drawContours(imgContour, contours, -1, 255, 2)
    cv2.drawContours(imgContour, [cnt], 0, (255), 1)
    cv2.drawContours(img_contour_text, [cnt], 0, (255), 1)
    moments = cv2.moments(cnt, False)
    x_coord = moments['m10'] / moments['m00']
    y_coord = moments['m01'] / moments['m00']
    cv2.putText(img_contour_text, f'area:{areas[max_index]}', (int(x_coord), int(y_coord)),
                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255),
                thickness=2)
    cv2.imshow("square", imgContour)
    return imgContour, cnt


def find_corners(img):
    img_corners = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))
    dst = cv.cornerHarris(img, 5, 9, 0.14)
    moments = cv2.moments(img, True)
    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img_corners[dst > 0.1 * dst.max()] = [0, 0, 255]
    x_coord = moments['m10'] / moments['m00']
    y_coord = moments['m01'] / moments['m00']
    img_corners[int(x_coord), int(y_coord)] = [0, 0, 255]
    cv2.drawMarker(img_corners, (int(x_coord), int(y_coord)), thickness=2, markerSize=2, markerType=cv2.MARKER_DIAMOND,
                   line_type=cv2.LINE_4, color=(0, 0, 255))
    cv2.imshow("corners", img_corners)


def image_handler():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    while True:
        ret, frame = cap.read()
        cv.imshow("frame", frame)
        img_blue = highlight_color(frame)
        img_blur = cv.GaussianBlur(img_blue, (3, 3), 1, 1, cv2.BORDER_DEFAULT)
        img, cnt = find_contours(img_blur)
        find_lines(img)
        # find_corners(img)
        if cv.waitKey(1) == 27:
            break
