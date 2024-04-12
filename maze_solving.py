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
                img_transformed = cv2.circle(img_transformed, (int(j * step), int(i * step)), 2, thickness=1,
                                             color=(0, 255, 0))
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
            return output
        except:
            print("exception getting output coord")
            return None

    def get_solution_path(self, weight_path, start_point):
        exit = [start_point]
        point = start_point
        rows, cols = weight_path.shape
        try:
            while weight_path[point]:
                point_weight = weight_path[point]
                h, w = point
                if h > 0 and weight_path[h - 1, w] < point_weight and not (self.path_matrix[h, w] & self._top):
                    point = (h - 1, w)
                    exit.append(point)
                    continue
                if h < rows - 2 and weight_path[h + 1, w] < point_weight and not (self.path_matrix[h, w] & self._bottom):
                    point = (h + 1, w)
                    exit.append(point)
                    continue
                if w > 0 and weight_path[h, w - 1] < point_weight and not (self.path_matrix[h, w] & self._left):
                    point = (h, w - 1)
                    exit.append(point)
                    continue
                if w < cols - 2 and weight_path[h, w + 1] < point_weight and not (self.path_matrix[h, w] & self._right):
                    point = (h, w + 1)
                    exit.append(point)
                    continue
                return None
            return exit
        except:
            print("getting solution path exception")
            return None

    def draw_weight_matrix(self, weight_matrix, length, image=None):
        if image is None:
            image = np.zeros((length, length), np.uint8)
        step_ver = image.shape[0] / weight_matrix.shape[0]
        step_hor = image.shape[1] / weight_matrix.shape[1]
        for i in range(0, weight_matrix.shape[0]):
            for j in range(0, weight_matrix.shape[1]):
                image = cv2.putText(image, str(weight_matrix[i, j]), thickness=1, color=(255, 255, 0),
                                    org=(round(j * step_hor), round((i + 1) * step_ver)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2)
        cv2.imshow("weight_matrix", image)

    def get_weight_matrix(self, entry_point):

        _path_matrix = self.path_matrix.copy()
        weight_matrix = np.full(_path_matrix.shape, None)
        points_with_same_weights = [entry_point]
        weight_matrix[entry_point[0], entry_point[1]] = 0
        new_points = []
        i = 1
        while True:
            for h, w in points_with_same_weights:
                if (not _path_matrix[h, w] & self._top) and weight_matrix[h - 1, w] is None:
                    weight_matrix[h - 1, w] = i
                    new_points.append((h - 1, w))
                if (not _path_matrix[h, w] & self._bottom) and weight_matrix[h + 1, w] is None:
                    weight_matrix[h + 1, w] = i
                    new_points.append((h + 1, w))
                if (not _path_matrix[h, w] & self._left) and weight_matrix[h, w - 1] is None:
                    weight_matrix[h, w - 1] = i
                    new_points.append((h, w - 1))
                if (not _path_matrix[h, w] & self._right) and weight_matrix[h, w + 1] is None:
                    weight_matrix[h, w + 1] = i
                    new_points.append((h, w + 1))
            if not new_points:
                break
            points_with_same_weights = new_points.copy()
            new_points.clear()
            i += 1
        return weight_matrix

    def pathHighlight(self, img, path):
        size = img.shape[0] / self.block_count
        for coordinate in path:
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
        mask_yellow = cv2.inRange(img_hsv, (20, 40, 142), (40, 180, 230))
        img_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
        cv2.imwrite("img_hsv.png",img_hsv)
        cv2.imshow("img_yellow", img_yellow)
        img_bin = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)
        img_open = cv2.morphologyEx(img_bin, anchor=(0, 0), kernel=np.ones((3, 3), np.uint8), iterations=4, op=cv2.MORPH_OPEN)
        print(f"img_bin {img_bin.shape}")
        print(f"img_open {img_open.shape}")
        cv2.imshow("img_open", img_open)
        img_canny = cv2.Canny(img_bin, 70, 210)
        cv2.imshow("canny without circles", img_canny)
        circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT, 1, 30, param1=20, param2=8, minRadius=8, maxRadius=20)
        if len(circles[0]) != 5:
            print(f"Найдено {len(circles[0])}  окружностей")
            return None, None
        print(f"img_open {img_open.shape}")
        circles_filtered_by_radius = sorted(circles[0], key=lambda x: x[2])
        circles_coord = [(x[0], x[1]) for x in circles_filtered_by_radius]
        ball_coord = circles_filtered_by_radius[0]
        corners_coord = circles_coord[-4:]
        x_sort = sorted(corners_coord, key=lambda x: x[0])
        left_sorted = sorted(x_sort[:2], key=lambda x: x[1])
        right_sorted = sorted(x_sort[2:], key=lambda x: x[1])
        initial_points = [left_sorted[0], left_sorted[1], right_sorted[1], right_sorted[0]]
        print(circles)
        circles_arounded = np.uint16(np.around(circles))
        for i in circles_arounded[0, :]:
            # draw the outer circle
            cv2.circle(img_canny, (i[0], i[1]), i[2], 120, 2)
            # draw the center of the circle
            cv2.circle(img_canny, (i[0], i[1]), 2, 120, 3)
        # cv2.imshow("img_canny", img_canny)
        # cv2.imshow("img",img)
        print(f"img {img.shape}")
        M = cv2.getPerspectiveTransform(np.float32(initial_points), np.float32([[0, 0], [0, self.aligned_img_len], [self.aligned_img_len, self.aligned_img_len], [self.aligned_img_len, 0]]))
        img_aligned = cv2.warpPerspective(img, M, (self.aligned_img_len, self.aligned_img_len))
        ball_coord = np.float32(np.array([[[ball_coord[0], ball_coord[1]]]]))
        ball_coord_transf = cv2.perspectiveTransform(ball_coord, M)[0]
        cv2.imshow("img_aligned",img_aligned)
        return img_aligned, ball_coord_transf[0]


def maze_solver():
    cap = cv2.VideoCapture(1)
    maze = Maze()

    while True:
        cv2.waitKey(1)
        cap.read()
        ret, frame = cap.read()
        cv2.imshow(" org", frame)
        cv2.imwrite("frame.png", frame)
        start = round(time.time() * 1000)
        img_aligned, ball_position = maze.flatten_table(frame)
        if img_aligned is None:
            continue
        maze.build_path_matrix(img_aligned.copy())
        img_grid = maze.draw_grid(img_aligned.copy())
        cv2.imshow("grid",img_grid)
        img_maze = maze.draw_path_matrix()
        cv2.imshow("maze", img_maze)
        out_point = maze.find_output_coordinates()
        if out_point is None:
            print("Выходы не найдены/Их несколько")
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



    # edgesToSolve = closeEntryPoint(point1, edges)
    # path = solveMethod(img, [point1, point2], edgesToSolve)
    # pathImg = pathHighlight(img, path)
    # cv2.imshow("path", pathImg)


maze_solver()
