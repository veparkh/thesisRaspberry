import time

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
    # cv.imshow("img_thresh", img_thresh)
    img_contours = np.zeros_like(img_thresh)
    img_erosed = cv.erode(img_thresh, kernel=np.ones((3, 3), np.uint8), anchor=(0, 0), iterations=1)
    # cv.imshow("img_erosed", img_erosed)

    contours, hierarchy = cv.findContours(img_erosed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_contours, contours=contours, hierarchy=hierarchy, contourIdx=-1, color=255)
    # cv.imshow("img_contours", img_contours)
    if len(contours) != 0:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        return img[y - 10:y + h + 10, x - 10:x + w + 10]
    return None


def flatten_table(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imwrite("img_hsv.png", img_hsv)
    lower_red = np.array([0, 30, 120])
    upper_red = np.array([10, 200, 210])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([150, 40, 120])
    upper_red = np.array([180, 200, 200])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # joinmasks
    mask = mask0 + mask1
    cv2.imshow("img_mask", mask)

    img_red = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("img_red", img_red)
    img_bin = cv.cvtColor(img_red, cv.COLOR_BGR2GRAY)
    img_canny = cv.Canny(img_bin, 70, 210)
    cv2.imshow("canny without circles", img_canny)

    circles = cv.HoughCircles(img_canny, cv.HOUGH_GRADIENT, 1, 10,
                              param1=20, param2=7, minRadius=3, maxRadius=10)

    circles = np.uint16(np.around(circles))
    thresh, img_thresh = cv.threshold(img_bin, 40, 255, cv.THRESH_BINARY)
    circles_sorted_by_rad = sorted(circles, key=lambda x: x[2])
    ball_coord = circles_sorted_by_rad[0][-1][:-1]

    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img_canny, (i[0], i[1]), i[2], 120, 2)
        # draw the center of the circle
        cv.circle(img_canny, (i[0], i[1]), 2, 120, 3)
    circles_filtered_by_rad = list(filter(lambda x: (x[2] > 5), circles[0,:]))

    # print(len(circles[0]))
    cv.imshow("img_thresh ", img_thresh)
    cv.imshow("img_canny", img_canny)
    if len(circles_filtered_by_rad) < 4:
        return None, None
    circles_coord = [(x[0],x[1]) for x in circles_filtered_by_rad]
    x_sort = sorted(circles_coord, key=lambda x: x[0])
    circles_coord = x_sort[:2] + x_sort[-2:]
    x_sort = sorted(circles_coord, key=lambda x: x[0])
    left_sorted = sorted(x_sort[:2], key=lambda x: x[1])
    right_sorted = sorted(x_sort[2:], key=lambda x: x[1])
    initial_points = [left_sorted[0], left_sorted[1], right_sorted[1], right_sorted[0]]
    square_len = 376
    M = cv.getPerspectiveTransform(np.float32(initial_points), np.float32([[0, 0], [0, square_len], [square_len, square_len], [square_len, 0]]))
    img_aligned = cv.warpPerspective(img, M, (square_len, square_len))
    ball_coord = np.float32(np.array([[[ball_coord[0], ball_coord[1]]]]))
    ball_coord_transf = cv2.perspectiveTransform(ball_coord, M)[0]
    return img_aligned, ball_coord_transf[0]


def maze_solver():
    cap = cv.VideoCapture(1)
    maze = maze_solving.Maze()

    while True:
        ret, frame = cap.read()
        try:
            cv.imshow(" org", frame)
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imwrite("frame.png", img_hsv)
            start = round(time.time() * 1000)
            img_aligned, ball_position = flatten_table(frame)
            if img_aligned is not None:
                maze.get_path_matrix(img_aligned)

                # maze.draw_grid(img_aligned)
                # maze.draw_path_matrix()
                out_point = maze.find_output_coordinates()
                if out_point is None:
                    print("out_point is  None")
                    continue
                weight_matrix = maze.get_weight_matrix(out_point[0])
                # maze.draw_ball(img_aligned, ball_position)
                # maze.draw_weight_matrix(weight_matrix, 0, img_aligned.copy())
                ball_aligned_position = maze.get_ball_position(ball_position, img_aligned.shape)
                if ball_aligned_position is None:
                    print("Шарик не определён")
                    continue
                print("ball pos", ball_aligned_position)
                exit_path = maze.get_solution_path(weight_matrix, ball_aligned_position)
                if exit_path is not None:
                    img_solved = maze.pathHighlight(img_aligned.copy(), exit_path)
                    cv2.imshow("img_solved", img_solved)
            end = round(time.time() * 1000)
            print(end - start)
            cv2.waitKey(1)
        except:
            print("exception")

    # edgesToSolve = closeEntryPoint(point1, edges)
    # path = solveMethod(img, [point1, point2], edgesToSolve)
    # pathImg = pathHighlight(img, path)
    # cv.imshow("path", pathImg)


maze_solver()
