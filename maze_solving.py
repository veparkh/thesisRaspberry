import time

import cv2
import numpy as np


class Maze:
    centres_count = 19
    block_count = centres_count - 1
    aligned_img_len = 376  # Размер по x и y
    _top = 0b1
    _bottom = 0b10
    _left = 0b100
    _right = 0b1000

    _PATH_MATRIX_PATTERN = np.array(
        [[_top | _left, _top, _top, _top, _top, _top, _top, _top, _top, _top, _top, _top, _top, _top, _top, _top, _top, _top | _right],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _right, ],
         [_left | _bottom, _bottom, _bottom, _bottom, _bottom, _bottom, _bottom, _bottom, _bottom, _bottom, _bottom, _bottom, _bottom, _bottom,
          _bottom, _bottom, _bottom, _bottom | _right]],
        np.uint8)

    exit_path = None
    exit_point = None
    weight_matrix = None
    path_matrix = None
    block_size = aligned_img_len / block_count

    def build_path_matrix(self, img, threshold=0.8, indent=6, width=3):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        step = self.aligned_img_len / self.block_count
        self.path_matrix = self._PATH_MATRIX_PATTERN.copy()
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh_path", img_thresh)
        for i in range(1, self.block_count - 1):
            for j in range(1, self.block_count - 1):

                left_block = img_thresh[round(step * i) + indent:round(step * (i + 1)) - indent, round(step * j) - width:round(step * j) + width]
                if cv2.countNonZero(left_block) / left_block.size < threshold:
                    self.path_matrix[i, j] |= self._left
                top_block = img_thresh[round(step * i) - width:round(step * i) + width, round(step * j) + indent:round(step * (j + 1)) - indent]
                if cv2.countNonZero(top_block) / top_block.size < threshold:
                    self.path_matrix[i, j] |= self._top

                right_block = img_thresh[round(step * i) + indent:round(step * (i + 1)) - indent,
                              round(step * (j + 1)) - width:round(step * (j + 1)) + width]
                if cv2.countNonZero(right_block) / right_block.size < threshold:
                    self.path_matrix[i, j] |= self._right

                bottom_block = img_thresh[round(step * (i + 1) - width):round(step * (i + 1) + width),
                               round(step * j) + indent:round(step * (j + 1)) - indent]
                if cv2.countNonZero(bottom_block) / bottom_block.size < threshold:
                    self.path_matrix[i, j] |= self._bottom

    def draw_grid(self, img_transformed):
        step = img_transformed.shape[0] / self.block_count
        for i in range(0, self.centres_count):
            for j in range(0, self.centres_count):
                img_transformed = cv2.circle(img_transformed, (int(j * step), int(i * step)), 2, thickness=1, color=(124))
        return img_transformed

    def draw_path_matrix(self):
        step = self.aligned_img_len / self.block_count
        if self.path_matrix is None:
            return None
        img_path_matrix = np.zeros((self.aligned_img_len, self.aligned_img_len), np.uint8)
        for i in range(1, self.block_count - 1):
            for j in range(1, self.block_count - 1):
                if self.path_matrix[i, j] & self._top:
                    img_path_matrix[round(step * i) - 3:round(step * i) + 3, round(step * j) + 3:round(step * (j + 1))] = 255
                if self.path_matrix[i, j] & self._bottom:
                    img_path_matrix[round(step * (i + 1) - 3):round(step * (i + 1) + 3), round(step * j):round(step * (j + 1))] = 255
                if self.path_matrix[i, j] & self._left:
                    img_path_matrix[round(step * i):round(step * (i + 1)), round(step * j) - 3:round(step * j) + 3] = 255
                if self.path_matrix[i, j] & self._right:
                    img_path_matrix[round(step * i):round(step * (i + 1)), round(step * (j + 1)) - 3:round(step * (j + 1)) + 3] = 255
        return img_path_matrix

    def find_output_coordinates(self):
        zero_shift_left = 1
        zero_shift_right = 1
        zero_shift_top = 1
        zero_shift_bottom = 1
        path_matrix_without_edges = self.path_matrix[1:-1, 1:-1]
        cols = self.path_matrix.shape[1]
        rows = self.path_matrix.shape[0]

        try:
            for i in path_matrix_without_edges:
                if np.any(i):
                    break
                zero_shift_top += 1
            for i in path_matrix_without_edges[::-1]:
                if np.any(i):
                    break
                zero_shift_bottom += 1
            for i in path_matrix_without_edges.T:
                if np.any(i):
                    break
                zero_shift_left += 1
            for i in path_matrix_without_edges.T[::-1]:
                if np.any(i):
                    break
                zero_shift_right += 1

            pm_without_zeros_edges = self.path_matrix[zero_shift_top + 1:-(zero_shift_bottom + 1), zero_shift_left + 1:-(zero_shift_right + 1)]
            output = []
            is_side_open = [0, 0, 0, 0]
            for ind, val in enumerate(pm_without_zeros_edges[0]):
                if not val & self._top:
                    is_side_open[0] = 1
                    output.append([zero_shift_top, zero_shift_left + ind + 1])
            for ind, val in enumerate(pm_without_zeros_edges[-1]):
                if not val & self._bottom:
                    is_side_open[1] = 1
                    output.append([rows - zero_shift_bottom - 1, zero_shift_left + ind + 1])
            for ind, val in enumerate(pm_without_zeros_edges[:, 0]):
                if not val & self._left:
                    is_side_open[2] = 1
                    output.append([zero_shift_top + ind + 1, zero_shift_left])
            for ind, val in enumerate(pm_without_zeros_edges[:, -1]):
                if not val & self._right:
                    is_side_open[3] = 1
                    output.append([zero_shift_top + ind + 1, cols - zero_shift_right - 1])
            if sum(is_side_open) != 1:
                print("Открыто несколько сторон")
                return None
            if len(output) == 0:
                print("Выходы не найдены")
                return None
            self.exit_point = output
            return
        except:
            print("exception getting output coord")
            return None

    def get_solution_path(self, start_point):
        if self.weight_matrix is None:
            print("get_solution_path: weight_matrix is None")
            return None
        exit_points = [start_point]
        point = start_point
        rows, cols = self.weight_matrix.shape
        try:
            while self.weight_matrix[point]:
                point_weight = self.weight_matrix[point]
                h, w = point
                if h > 0 and self.weight_matrix[h - 1, w] < point_weight and not (self.path_matrix[h, w] & self._top):
                    point = (h - 1, w)
                    exit_points.append(point)
                    continue
                if h < rows - 2 and self.weight_matrix[h + 1, w] < point_weight and not (self.path_matrix[h, w] & self._bottom):
                    point = (h + 1, w)
                    exit_points.append(point)
                    continue
                if w > 0 and self.weight_matrix[h, w - 1] < point_weight and not (self.path_matrix[h, w] & self._left):
                    point = (h, w - 1)
                    exit_points.append(point)
                    continue
                if w < cols - 2 and self.weight_matrix[h, w + 1] < point_weight and not (self.path_matrix[h, w] & self._right):
                    point = (h, w + 1)
                    exit_points.append(point)
                    continue
                return False
            self.exit_path = exit_points
            return True
        except:
            print("getting solution path exception")
            return False

    def draw_weight_matrix(self, length, image=None):
        if self.weight_matrix is None:
            print("draw_weight_matrix: weight_matrix is None")
            return
        if image is None:
            image = np.zeros((length, length), np.uint8)
        step_ver = image.shape[0] / self.weight_matrix.shape[0]
        step_hor = image.shape[1] / self.weight_matrix.shape[1]
        for i in range(0, self.weight_matrix.shape[0]):
            for j in range(0, self.weight_matrix.shape[1]):
                image = cv2.putText(image, str(self.weight_matrix[i, j]), thickness=1, color=(255, 255, 0),
                                    org=(round(j * step_hor), round((i + 1) * step_ver)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2)
        cv2.imshow("weight_matrix", image)

    def get_weight_matrix(self):
        if self.exit_point is None:
            print("get_weight_matrix: out_point is None:")
            return False
        _path_matrix = self.path_matrix.copy()
        self.weight_matrix = np.full(_path_matrix.shape, None)
        points_with_same_weights = [self.exit_point[0]]
        self.weight_matrix[self.exit_point[0][0], self.exit_point[0][1]] = 0
        new_points = []
        i = 1
        while True:
            for h, w in points_with_same_weights:
                if (not _path_matrix[h, w] & self._top) and self.weight_matrix[h - 1, w] is None:
                    self.weight_matrix[h - 1, w] = i
                    new_points.append((h - 1, w))
                if (not _path_matrix[h, w] & self._bottom) and self.weight_matrix[h + 1, w] is None:
                    self.weight_matrix[h + 1, w] = i
                    new_points.append((h + 1, w))
                if (not _path_matrix[h, w] & self._left) and self.weight_matrix[h, w - 1] is None:
                    self.weight_matrix[h, w - 1] = i
                    new_points.append((h, w - 1))
                if (not _path_matrix[h, w] & self._right) and self.weight_matrix[h, w + 1] is None:
                    self.weight_matrix[h, w + 1] = i
                    new_points.append((h, w + 1))
            if not new_points:
                break
            points_with_same_weights = new_points.copy()
            new_points.clear()
            i += 1
        return True

    def pathHighlight(self, img):
        if self.exit_path is None:
            return None
        size = img.shape[0] / self.block_count
        for coordinate in self.exit_path:
            h = round(size * (coordinate[0] + 1))
            w = round(size * (coordinate[1] + 1))
            h0 = round(size * coordinate[0])
            w0 = round(size * coordinate[1])
            img[h0:h, w0:w] = img[h0:h, w0:w] - 50
        return img

    def draw_ball(self, img_aligned, ball_coord):
        img_ball = cv2.circle(img_aligned, center=(round(ball_coord[0]), round(ball_coord[1])), thickness=2, color=(255, 0, 255), lineType=cv2.LINE_8,
                              radius=3)
        cv2.imshow("img_ball", img_ball)
        return img_aligned

    def get_ball_position(self, ball_coord, size):
        x = int(self.block_count * ball_coord[1] / size[1])
        y = int(self.block_count * ball_coord[0] / size[0])
        if x <= 0 or x >= self.block_count or y <= 0 or y >= self.block_count:
            print("Шарик за пределами лабиринта")
            return None
        return x, y

    def get_coordinates_by_position(self, position):
        return self.block_count * (0.5 + position[0]), self.block_count * (0.5 + position[1])

    def flatten_table(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, (17, 63, 149), (37, 249, 215))
        img_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
        img_bin = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_bin, 70, 210)
        circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT, 1, 30, param1=20, param2=8, minRadius=6, maxRadius=20)
        print(f"Найдено {len(circles[0])}  окружностей")
        if len(circles[0]) != 5:
            print(f"Найдено {len(circles[0])}  окружностей")
            return None, None
        circles_filtered_by_radius = sorted(circles[0], key=lambda x: x[2])
        circles_coord = [(x[0], x[1]) for x in circles_filtered_by_radius]
        ball_coord = circles_filtered_by_radius[0]
        corners_coord = circles_coord[-4:]
        x_sort = sorted(corners_coord, key=lambda x: x[0])
        left_sorted = sorted(x_sort[:2], key=lambda x: x[1])
        right_sorted = sorted(x_sort[2:], key=lambda x: x[1])
        initial_points = [left_sorted[0], left_sorted[1], right_sorted[1], right_sorted[0]]
        circles_arounded = np.uint16(np.around(circles))
        # for i in circles_arounded[0, :]:
        #     # draw the outer circle
        #     cv2.circle(img_canny, (i[0], i[1]), i[2], 120, 2)
        #     # draw the center of the circle
        #     cv2.circle(img_canny, (i[0], i[1]), 2, 120, 3)
        # # cv2.imshow("img_canny", img_canny)
        # # cv2.imshow("img",img)
        # cv2.imshow("img_canny_with_circles", img_canny)
        M = cv2.getPerspectiveTransform(np.float32(initial_points), np.float32(
            [[0, 0], [0, self.aligned_img_len], [self.aligned_img_len, self.aligned_img_len], [self.aligned_img_len, 0]]))
        img_aligned = cv2.warpPerspective(img, M, (self.aligned_img_len, self.aligned_img_len))
        ball_coord = np.float32(np.array([[[ball_coord[0], ball_coord[1]]]]))
        ball_coord_transf = cv2.perspectiveTransform(ball_coord, M)[0]
        # cv2.imshow("img_aligned", img_aligned)
        return img_aligned, ball_coord_transf[0]

    def solve_maze(self, img_orig, rebuild_solution: bool):
        img_table, ball_coord = self.flatten_table(img_orig)
        if img_table is None:
            return None, None, None, None, None
        if rebuild_solution or (self.weight_matrix is None) or (self.exit_point is None):
            self.build_path_matrix(img_table)
            self.find_output_coordinates()
        if self.exit_point is None:
            return None, None, None, None, img_table
        self.get_weight_matrix()
        ball_aligned_position = self.get_ball_position(ball_coord, img_table.shape)
        if ball_aligned_position is None:
            return None, None, None, None, img_table
        if self.exit_path is None or ball_aligned_position not in self.exit_path:
            if self.get_solution_path(ball_aligned_position):
                return None, None, None, None, img_table
        index = self.exit_path.index(ball_aligned_position)
        if len(self.exit_path) == index+1:
            return None, None, None, None, img_table
        print("ball aligned coord", ball_aligned_position,"next pos", self.exit_path[index + 1])
        return (self.exit_path[index + 1][1]+0.5)*self.block_size,(self.exit_path[index + 1][0]+0.5)*self.block_size, ball_coord[0], ball_coord[1], img_table


def maze_solver():
    maze = Maze()
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        start = time.time()
        x_goal, y_goal, x_curr, y_curr, img_table = maze.solve_maze(frame, False)
        print("x_goal x_curr, y_goal, y_curr", x_goal, x_curr, y_goal, y_curr)
        # img_maze = maze.draw_path_matrix()
        # if img_maze is None:
        #     continue
        # cv2.imshow("path_matrix",img_maze)
        if x_goal is None:
            continue
        img_solution = maze.pathHighlight(img_table)
        cv2.imshow("img_solution", img_solution)
        cv2.waitKey(1)

# maze_solver()