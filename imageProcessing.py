import cv2
import cv2 as cv
import numpy as np


def highlight_color(img, lower, upper):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
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


def findPathMatrix():
    labyrImg = cv2.imread("labyrinth3.png", cv2.IMREAD_GRAYSCALE)
    labyrImg = 255 - labyrImg
    cv2.imshow("orig", labyrImg)
    _, labyrImg = cv2.threshold(labyrImg, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", labyrImg)

    horLines, verLines = findBlockLines(labyrImg)

    # paths = np.zeros(shape=(len(horizontalLines), len(verticalLines)))
    findInputs(horLines, verLines, labyrImg)
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


def find_maze_rectangle(img):
    pass


def maze_solver():
    cap = cv.VideoCapture(1)
    ret, frame = cap.read()
    shape = frame.shape
    frame = frame[:, int(0.2 * shape[1]):int(0.8 * shape[1])]
    print()
    img_hsv = highlight_color(frame, np.array([0, 0, 0]), np.array([179, 200, 105]))
    img_cleared = cv2.morphologyEx(img_hsv, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imshow("walls_hsv_cleared", img_cleared)
    cv2.imwrite("roi.png", img_cleared)
    cv2.waitKey(0)


maze_solver()
