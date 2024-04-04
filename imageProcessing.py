import math

import cv2
import cv2 as cv
import numpy
import numpy as np
from cv2 import threshold


def highlight_color(img, lower, upper):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv2.imwrite("roi.png", img)
    mask = cv.inRange(hsv, lower, upper)
    return mask


def image_handler():
    cap = cv.VideoCapture(1)
    ret, frame = cap.read()
    while True:
        ret, frame = cap.read()
        cv.imshow("frame", frame)

        # find_corners(img)
        if cv.waitKey(1) == 27:
            break


def findBlockLines(img):
    lines = cv2.HoughLines(img, 0.1, np.pi / 280, 100, 0, 0)
    verticalLines = list(filter(lambda x: -1 < x[0][1] < 1, lines))
    verLinesFiltered = list()
    horLinesFiltered = list()

    horizontalLines = list(filter(lambda x: 1 < x[0][1] < 2, lines))
    verticalLines.sort(key=lambda x: x[0][0])
    horizontalLines.sort(key=lambda x: x[0][0])
    for i in range(0, len(verticalLines), 2):
        if verticalLines[i][0][0] - 2 < abs(verticalLines[i][0][0]) < verticalLines[i + 1][0][0] + 2:
            verLinesFiltered.append([(verticalLines[i][0][0] + verticalLines[i + 1][0][0]) / 2, 0])
    for i in range(0, len(horizontalLines), 2):
        if horizontalLines[i][0][0] - 2 < abs(horizontalLines[i][0][0]) < horizontalLines[i + 1][0][0] + 2:
            horLinesFiltered.append([(horizontalLines[i][0][0] + horizontalLines[i + 1][0][0]) / 2, np.pi / 2])
    print(verLinesFiltered)
    print(len(verLinesFiltered))
    print(horLinesFiltered)
    print(len(horLinesFiltered))
    alLines = horLinesFiltered + verLinesFiltered
    linesImg = np.zeros_like(img)
    if alLines is not None:
        for i in range(0, len(alLines)):
            rho = alLines[i][0]
            theta = alLines[i][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv.line(linesImg, pt1, pt2, (255), 1, cv.LINE_4)
    cv2.imshow("lines", linesImg)
    return horLinesFiltered, verLinesFiltered


def findPathMatrix(img):
    cv2.imshow("orig", img)
    _, labyrImg = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", img)
    cv2.waitKey(0)
    horLines, verLines = findBlockLines(img)

    # paths = np.zeros(shape=(len(horizontalLines), len(verticalLines)))
    findInputs(horLines, verLines, img)
    cv2.waitKey(0)


def findEdges(block):
    edge = 0
    topBlock = block[0:int(block.shape[0] / 2), :]
    leftBlock = block[:, 0:int(block.shape[1] / 2)]
    rightBlock = block[:, int(block.shape[1] / 2):block.shape[1]]
    bottomBlock = block[int(block.shape[0] / 2):block.shape[0], :]
    cv2.imshow("topBlock", topBlock)
    cv2.imshow("leftBlock", leftBlock)
    cv2.imshow("rightBlock", rightBlock)
    cv2.imshow("bottomBlock", bottomBlock)
    cv2.imshow("Block", block)
    line = cv2.HoughLines(topBlock, 0.2, np.pi / 180, 15)
    if line is None:
        edge += 1000
    line = cv2.HoughLines(bottomBlock, 0.2, np.pi / 180, 15)
    if line is None:
        edge += 100
    line = cv2.HoughLines(leftBlock, 0.2, np.pi / 180, 15)
    if line is None:
        edge += 10
    line = cv2.HoughLines(rightBlock, 0.2, np.pi / 180, 15)
    if line is None:
        edge += 1
    return edge


def findInputs(horLines, verLines, img):
    # верх`
    point1 = (0, 0)
    point2 = (0, 0)
    for i in range(0, len(verLines) - 1):
        cv2.waitKey(0)
        topBlock = img[int(horLines[0][0]):int(horLines[1][0] / 2), int(verLines[i][0]):int(verLines[i + 1][0] + 2)]
        line = cv2.HoughLines(topBlock, 0.2, np.pi / 180, 15)
        if (line is None):
            point1 = (0, i)
        print(line)
        cv2.imshow("topBlock", topBlock)
    # лево # право
    # низ
    for i in range(0, len(verLines) - 1):
        block = img[int(horLines[-2][0]):int(horLines[-1][0] + 1), int(verLines[i][0]):int(verLines[i + 1][0] + 2)]
        bottomBlock = img[
                      int((horLines[-2][0] + horLines[-1][0]) / 2):int(horLines[-1][0] + 2),
                      int(verLines[i][0]):int(verLines[i + 1][0] + 2)
                      ]
        line = cv2.HoughLines(bottomBlock, 0.2, np.pi / 180, 15)
        if (line is None):
            point2 = (len(horLines) - 2, i)
    print(point1)
    print(point2)

    edges = list()
    for i in range(0, len(horLines) - 1):
        rowEdges = list()
        for j in range(0, len(verLines) - 1):
            block = img[int(horLines[i][0]):int(horLines[i + 1][0] + 2),
                    int(verLines[j][0]):int(verLines[j + 1][0] + 2)]
            rowEdges.append(findEdges(block))
        edges.append(rowEdges)
    print(edges)
    edgesToSolve = closeEntryPoint(point1, edges)
    path = solveMethod(img, [point1, point2], edgesToSolve)
    pathImg = pathHighlight(img, path)
    cv2.imshow("path", pathImg)
    return [point1, point2], edges


def closeEntryPoint(entryPoint, edges):
    h, w = entryPoint[0], entryPoint[1]

    if h == 0:
        if (edges[h][w] >= 1000):
            edges[h][w] -= 1000
    if w == 0:
        edges[h][w] &= 0b1101
    if h == len(edges) - 1:
        edges[h][w] &= 0b1011
    if w == len(edges[0]) - 1:
        edges[h][w] &= 0b1110
    print(edges)
    return edges


def solveMethod(origImg, entryPoint, edges):
    shortestPath = []
    img = origImg
    sp = []
    rec = [0]
    p = 0
    sp.append(list(entryPoint[0]))
    print(entryPoint[0], entryPoint[1])
    print(sp[p][0], sp[p][1])
    cv2.waitKey(0)
    while True:
        h, w = sp[p][0], sp[p][1]
        print(h, w)
        # h stands for height and w stands for width
        if sp[-1] == list(entryPoint[1]):
            break
        if edges[h][w] > 0:
            rec.append(len(sp))
        if edges[h][w] > 999:
            # If this edges is open upwards
            edges[h][w] = edges[h][w] - 1000
            h = h - 1
            sp.append([h, w])
            edges[h][w] = edges[h][w] - 100
            p = p + 1
            continue
        if edges[h][w] > 99:
            # If the edges is open downward
            edges[h][w] = edges[h][w] - 100
            h = h + 1
            sp.append([h, w])
            edges[h][w] = edges[h][w] - 1000
            p = p + 1
            continue
        if edges[h][w] > 9:
            # If the edges is open left
            edges[h][w] = edges[h][w] - 10
            w = w - 1
            sp.append([h, w])
            edges[h][w] = edges[h][w] - 1
            p = p + 1
            continue
        if edges[h][w] == 1:
            # If the edges is open right
            edges[h][w] = edges[h][w] - 1
            w = w + 1
            sp.append([h, w])
            edges[h][w] = edges[h][w] - 10
            p = p + 1
            continue
        else:
            # Removing the coordinates that are closed or don't show any path
            sp.pop()
            rec.pop()
            p = rec[-1]

    for i in sp:
        shortestPath.append(tuple(i))
    print(shortestPath)
    return shortestPath


def pathHighlight(img, path):
    size = 16
    for coordinate in path:
        h = size * (coordinate[0] + 1)
        w = size * (coordinate[1] + 1)
        h0 = size * coordinate[0]
        w0 = size * coordinate[1]
        img[h0:h, w0:w] = img[h0:h, w0:w] - 50
    return img


def detect_maze_coarse_borders(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite("maze_hsv.jpeg", img_hsv)
    img_lines = np.zeros_like(img_hsv)
    lower_red = np.array([0, 20, 60], dtype="uint8")
    upper_red = np.array([140, 90, 100], dtype="uint8")
    img_borders = cv2.inRange(img_hsv, lower_red, upper_red)
    img_dilated = cv2.dilate(img_borders,borderType=cv2.BORDER_REPLICATE,anchor=(0,0),kernel=np.ones((3,3),np.uint8))
    img_closed = cv2.morphologyEx(img_borders,borderType=cv2.BORDER_DEFAULT,kernel=np.ones((3,3), dtype=np.uint8),anchor=(0,0),iterations=4,op = cv2.MORPH_CLOSE)
    cv2.imshow("closed", img_closed)
    cv2.imshow("dilated", img_dilated)
    img_Canny = cv2.Canny(img_closed,threshold1=100,threshold2=200)
    cv2.imshow("canny", img_Canny)
    cv2.imshow("im_bound",img_borders)
    cv2.waitKey(0)
    left = 20000
    right = 0
    bottom = 0
    top = 20000
    lines = cv.HoughLines(img_Canny, 1, np.pi / 180, 70, None, 0, 0)
    print(f"len size {len(lines)}")
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            print(f"x0 {x0} y0 {y0} theta {theta}")
            if -0.2 <= theta <= 0.2:
                if right < x0:
                    right = x0
                if left > x0:
                    left = x0
            elif 1.47 <= theta <= 1.67:
                if bottom < y0:
                    bottom = y0
                if top > y0:
                    top = y0
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(img_lines, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
        print(f"bottom {bottom}  top {top} left {left} right {right}")
        cv2.imshow("lines",img_lines)
        cv2.waitKey(0)
        return img[int(top-40):int(bottom+40),int(left-40):int(right+40)]

def get_path_matrix(img):

    step = 3
    vertical_lines_amount = round(img.shape[0]/step)
    horizontal_lines_amount = round(img.shape[1] / step)
    A = np.zeros((vertical_lines_amount,horizontal_lines_amount), np.uint8)
    for i in range(0, vertical_lines_amount-1):
        for j in range(0, horizontal_lines_amount-1):
            block = img[i*step:(i+1)*step,j*step:(j+1)*step]
            print(cv2.countNonZero(block))
            if cv2.countNonZero(block)/(step * step)>0.5:
                A[i,j] = 255
    cv2.imshow("A",A)
    cv2.imwrite("A.png",A)
    cv2.waitKey(0)


def detect_maze_fine_boarders(img):

    img_binary =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img_thresh = cv2.threshold(img_binary, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("thresh.png", img_thresh)
    img_closed = cv2.morphologyEx(img_thresh, borderType=cv2.BORDER_DEFAULT, kernel=np.ones((1, 3), dtype=np.uint8),
                                  anchor=(0, 0), iterations=4, op=cv2.MORPH_CLOSE)
    cv2.imshow("thresh",img_thresh)
    cv2.imshow("closed",img_closed)

    img_canny = cv2.Canny(img_thresh,100,200,3)
    cv2.imshow("img_canny", img_canny)
    cv2.waitKey(0)
    # lines = cv.HoughLines(img_canny, 1, np.pi / 180, 190, None, 0, 0)
    img_ver_lines = np.copy(img_canny)
    img_hor_lines = np.copy(img_canny)
    # left_1 = 1000
    # left_2 = 1000
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #         if left_1>x0:
    #             left_2 = left_1
    #             left_1 = x0
    #         if left_2>x0 and left_1!=x0:
    #             left_2 = x0
    #         cv.line(img_hor_lines, pt1, pt2, (255), 1, cv.LINE_AA)
    #
    # top_1 = 1000
    # top_2 = 1000
    # lines = cv.HoughLines(img_canny, 1, np.pi / 180, 70, None, 0, 0)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #         if 1.47<theta<1.67:
    #             if top_1>y0:
    #                 top_2 = top_1
    #                 top_1 = y0
    #             if top_2>y0 and (abs(top_1-y0)>5):
    #                 top_2= y0
    #             cv.line(img_ver_lines, pt1, pt2, (255), 1, cv.LINE_AA)
    #
    # print("point", int((top_1+top_2)/2),int((left_1+left_2)/2))
    # img_marker = cv2.drawMarker(img_canny,( int((left_1 + left_2) / 2),int((top_1 + top_2) / 2)),255)
    #
    # cv2.imshow("img_ver_lines", img_ver_lines)
    # cv2.imshow("img_hor_lines", img_hor_lines)
    cv2.imshow("canny",img_canny)
    lines = cv.HoughLines(img_canny, 1, np.pi / 180, 100, None, 0, 0)
    theta_aver = 0
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            if -0.3<theta<0.3:
                theta_aver += theta/len(lines)
    print(f"theta_aver {theta_aver}")
    M = cv2.getRotationMatrix2D((img_canny.shape[1] / 2, img_canny.shape[0] / 2), theta_aver*180/3.1415, 1)
    img_rotated = cv2.warpAffine(img_thresh, M, (img_canny.shape[1], img_canny.shape[0]))
    cv2.imshow("rotated",img_rotated)
    cv2.waitKey(0)
    img_final_cut = np.copy(img_rotated)
    rows,cols = img_final_cut.shape
    for i in range(0,cols):
        slice_left = img_final_cut[:,i]
        if cv2.countNonZero(slice_left)>0.7*rows:
            img_final_cut = img_final_cut[:,i+1:cols]
            break
    rows, cols = img_final_cut.shape
    for i in range(1,cols):
        slice_right = img_final_cut[:,cols-i]
        if cv2.countNonZero(slice_right)>0.7*rows:
            img_final_cut = img_final_cut[:,0:cols-(i+1)]
            break
    rows, cols = img_final_cut.shape
    for i in range(1,rows):
        slice_top = img_final_cut[i,:]
        if cv2.countNonZero(slice_top)>0.7*cols:
            img_final_cut = img_final_cut[i+1:rows,:]
            break
    rows, cols = img_final_cut.shape
    for i in range(1,rows):
        slice_bottom = img_final_cut[rows-i,:]
        if cv2.countNonZero(slice_bottom)>0.7*cols:
            img_final_cut = img_final_cut[0:rows-(i+1),:]
            break
    cv2.imshow("cut_final_img",img_final_cut)
    cv2.imwrite("fime_borders.png",img_final_cut)
    return img_final_cut

def detect_table(img):
    img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_binary,160,255,cv2.THRESH_BINARY)
    cv2.imshow("img_thresh",img_thresh)
    img_contours = np.zeros_like(img_thresh)
    img_erosed = cv2.erode(img_thresh,kernel=np.ones((3, 3), np.uint8),anchor=(0,0),iterations=1)
    cv2.imshow("img_erosed", img_erosed)

    contours, hierarchy = cv.findContours(img_erosed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contours, contours=contours, hierarchy=hierarchy, contourIdx=-1, color=255)
    cv2.imshow("img_contours",img_contours)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y-10:y + h+10, x-10:x + w+10]
    return None

def detect_maze_boarders(img):
    img_blurred = cv2.GaussianBlur(img,ksize=(5,5),sigmaX=1,sigmaY=1)
    img_bin = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    thtesh,img_thresh = cv2.threshold(img_bin,100,255, cv2.THRESH_OTSU)
    print(f"thresh {thtesh}")
    cv2.imshow("imgfd",img_thresh)
    img_lines = np.zeros_like(img_thresh)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img_thresh)[0]  # Position 0 of the returned tuple are the detected lines
    drawn_img = numpy.zeros_like(img_lines)
    drawn_img = lsd.drawSegments(drawn_img, lines)
    cv2.imshow("LSD", drawn_img)
    pass

def flatten_table(img):
    cv2.imshow("flatten_table", img)
    img_blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)
    cv2.imwrite("img_blurred.png",img_blurred)
    cv2.imshow("img_blurred", img_blurred)
    img_bin = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_bin,70,210)
    cv2.imshow("img_canny", img_canny)
    circles = cv.HoughCircles(img_canny, cv.HOUGH_GRADIENT, 1, 80,
                              param1=20, param2=7, minRadius=5, maxRadius=10)
    print(circles)
    circles = np.uint16(np.around(circles))
    thresh, img_thresh = cv2.threshold(img_bin, 40, 255, cv2.THRESH_BINARY)

    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img_thresh, (i[0], i[1]), i[2], 120, 2)
        # draw the center of the circle
        cv.circle(img_thresh, (i[0], i[1]), 2, 120, 3)

    print(len(circles[0]))
    cv2.imshow("img_thresh ", img_thresh)
    circles_coord = [[x[0], x[1]] for x in circles[0, :]]
    if len(circles_coord) < 4:
        cv2.waitKey(0)
    if len(circles_coord)>4:
        x_sort = sorted(circles_coord, key=lambda x: x[0])
        circles_coord = x_sort[:2] + x_sort[-2:]
    x_sort = sorted(circles_coord, key=lambda x: x[0])
    left_sorted = sorted(x_sort[:2], key=lambda x: x[1])
    right_sorted = sorted(x_sort[2:], key=lambda x: x[1])
    initial_points = [left_sorted[0], left_sorted[1],right_sorted[1],right_sorted[0]]# left_top, bottom-top,right_bottom
    print(f"initial points {initial_points}")
    M = cv.getPerspectiveTransform(np.float32(initial_points), np.float32([[0, 376], [0, 0], [376, 0], [376, 376]]))
    img_aligned = cv.warpPerspective(img, M, (376, 376))
    cv2.imshow("img_aligned", img_aligned)
    cv2.imwrite("img_aligned.png", img_aligned)
    img_grid = np.copy(img_aligned)
    n = 18
    step = 376/18
    for i in range(0, n+1):
        for j in range(0, n+1):
            img_grid = cv2.circle(img_grid,(int(j*step),int(i*step)),2,thickness=1,color=(0,255,0))
    cv2.imshow("grid", img_grid)







def maze_solver():
    cap = cv.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        cv2.imshow(" org", frame)
        img_table = detect_table(frame)
        if img_table is not None:
            flatten_table(img_table)
            # img_maze = detect_maze_boarders(img_table)

        cv2.waitKey(30)
    # img_cropped = frame[:, int(frame.shape[1] * 0.16):int(frame.shape[1] * 0.84)]
    #
    # cv2.imshow(" cropped img", img_cropped)
    # img_blured = cv2.bilateralFilter(img_cropped,d=5,sigmaSpace=20,sigmaColor=20)
    # cv2.imshow("cropped and blured",img_blured)
    # img_coarse = detect_maze_coarse_borders(img_blured)
    # img = detect_maze_fine_boarders(img_coarse)
    # cv2.imshow(" cropped img", img)
    # cv2.waitKey(0)
    # A = get_path_matrix(img)



maze_solver()
