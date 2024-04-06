import cv2
import cv2 as cv
import numpy as np
import maze_solving

def image_handler():
    cap = cv.VideoCapture(1)
    ret, frame = cap.read()
    while True:
        ret, frame = cap.read()
        cv.imshow("frame", frame)

        # find_corners(img)
        if cv.waitKey(1) == 27:
            break

def detect_table(img):
    img_binary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img_thresh = cv.threshold(img_binary, 160, 255, cv.THRESH_BINARY)
    cv.imshow("img_thresh", img_thresh)
    img_contours = np.zeros_like(img_thresh)
    img_erosed = cv.erode(img_thresh, kernel=np.ones((3, 3), np.uint8), anchor=(0, 0), iterations=1)
    cv.imshow("img_erosed", img_erosed)

    contours, hierarchy = cv.findContours(img_erosed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_contours, contours=contours, hierarchy=hierarchy, contourIdx=-1, color=255)
    cv.imshow("img_contours", img_contours)
    if len(contours) != 0:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        return img[y - 10:y + h + 10, x - 10:x + w + 10]
    return None

def flatten_table(img):
    cv.imshow("flatten_table", img)
    img_blurred = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)
    cv.imwrite("img_blurred.png", img_blurred)
    cv.imshow("img_blurred", img_blurred)
    img_bin = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)
    img_canny = cv.Canny(img_bin, 70, 210)
    cv.imshow("img_canny", img_canny)
    circles = cv.HoughCircles(img_canny, cv.HOUGH_GRADIENT, 1, 80,
                              param1=20, param2=7, minRadius=5, maxRadius=10)
    print(circles)
    circles = np.uint16(np.around(circles))
    thresh, img_thresh = cv.threshold(img_bin, 40, 255, cv.THRESH_BINARY)

    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img_thresh, (i[0], i[1]), i[2], 120, 2)
        # draw the center of the circle
        cv.circle(img_thresh, (i[0], i[1]), 2, 120, 3)

    print(len(circles[0]))
    cv.imshow("img_thresh ", img_thresh)
    circles_coord = [[x[0], x[1]] for x in circles[0, :]]
    if len(circles_coord) < 4:
        cv.waitKey(0)
    if len(circles_coord) > 4:
        x_sort = sorted(circles_coord, key=lambda x: x[0])
        circles_coord = x_sort[:2] + x_sort[-2:]
    x_sort = sorted(circles_coord, key=lambda x: x[0])
    left_sorted = sorted(x_sort[:2], key=lambda x: x[1])
    right_sorted = sorted(x_sort[2:], key=lambda x: x[1])
    initial_points = [left_sorted[0], left_sorted[1], right_sorted[1],
                      right_sorted[0]]  # left_top, bottom-top,right_bottom
    print(f"initial points {initial_points}")
    square_len = 376
    M = cv.getPerspectiveTransform(np.float32(initial_points),
                                   np.float32([[0, square_len], [0, 0], [square_len, 0], [square_len, square_len]]))
    img_aligned = cv.warpPerspective(img, M, (square_len, square_len))
    cv.imshow("img_aligned", img_aligned)
    cv.imwrite("img_aligned.png", img_aligned)
    img_grid = np.copy(img_aligned)
    return img_aligned


def maze_solver():
    cap = cv.VideoCapture(1)
    maze = maze_solving.Maze()
    while True:
        ret, frame = cap.read()
        cv.imshow(" org", frame)
        img_table = detect_table(frame)
        if img_table is not None:
            img_aligned = flatten_table(img_table)
            if img_aligned is not None:
                maze.get_path_matrix(img_aligned)
                maze.draw_grid(img_aligned)
                maze.draw_path_matrix()
                out_point = maze.find_output_coordinates()
                shortestPath = maze.solveMethod(img_aligned,[(8,11),out_point])
                solve = maze.pathHighlight(img_aligned, shortestPath)
                cv2.imshow("img_solved", solve)
                cv2.waitKey(0)
        cv.waitKey(30)
    # edgesToSolve = closeEntryPoint(point1, edges)
    # path = solveMethod(img, [point1, point2], edgesToSolve)
    # pathImg = pathHighlight(img, path)
    # cv.imshow("path", pathImg)


maze_solver()
