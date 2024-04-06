import cv2
import numpy as np


class Maze:
    path_matrix = None
    centres_count = 19
    lines_count = centres_count - 1

    _top = 0b1
    _bottom = 0b10
    _left = 0b100
    _right = 0b1000

    def get_path_matrix(self, img, threshold=0.8, indent=5, width=3):
        step = img.shape[0] / self.lines_count
        self.path_matrix = np.zeros((self.lines_count, self.lines_count), np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow("img_thresh", img_thresh)
        for i in range(1, self.lines_count - 1):
            for j in range(1, self.lines_count - 1):
                block = img_thresh[round(step * i):round(step * (i + 1)), round(step * j):round(step * (j + 1))]

                img[round(step * i) + indent:round(step * (i + 1)) - indent, round(step * j) - width:round(step * j) + width] = (255, 0, 0)
                left_block = img_thresh[round(step * i) + indent:round(step * (i + 1)) - indent, round(step * j) - width:round(step * j) + width]
                if cv2.countNonZero(left_block) / left_block.size < threshold:
                    self.path_matrix[i, j] |= self._left
                img[round(step * i) - width:round(step * i) + width, round(step * j) + indent:round(step * (j + 1)) - indent] = (0, 255, 0)
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
                cv2.imshow("bottom_block,", bottom_block)
                cv2.imshow("right_block,", right_block)
                cv2.imshow("top_block,", top_block)
                cv2.imshow("left_block,", left_block)
                cv2.imshow("block", block)
                cv2.imshow("img", img)

    def draw_grid(self, img_transformed):
        step = img_transformed.shape[0] / self.lines_count
        for i in range(0, self.centres_count):
            for j in range(0, self.centres_count):
                img_grid = cv2.circle(img_transformed, (int(j * step), int(i * step)), 2, thickness=1,
                                      color=(0, 255, 0))
        cv2.imshow("grid", img_transformed)

    def draw_path_matrix(self, img_len=376):
        step = img_len / self.lines_count
        if self.path_matrix is None:
            return None
        img_path_matrix = np.zeros((img_len, img_len), np.uint8)
        print("path", self.path_matrix)
        for i in range(1, self.lines_count - 1):
            for j in range(1, self.lines_count - 1):
                if self.path_matrix[i, j] & self._top:
                    img_path_matrix[round(step * i) - 3:round(step * i) + 3, round(step * j) + 3:round(step * (j + 1))] = 255
                if self.path_matrix[i, j] & self._bottom:
                    img_path_matrix[round(step * (i + 1) - 3):round(step * (i + 1) + 3), round(step * j):round(step * (j + 1))] = 255
                if self.path_matrix[i, j] & self._left:
                    img_path_matrix[round(step * i):round(step * (i + 1)), round(step * j) - 3:round(step * j) + 3] = 255
                if self.path_matrix[i, j] & self._right:
                    img_path_matrix[round(step * i):round(step * (i + 1)), round(step * (j + 1)) - 3:round(step * (j + 1)) + 3] = 255
        cv2.imshow("path)matrix", img_path_matrix)

    def find_output_coordinates(self):
        zero_shift_left = 0
        zero_shift_right = 0
        zero_shift_top = 0
        zero_shift_bottom = 0
        cols = self.path_matrix.shape[1]
        rows = self.path_matrix.shape[0]
        for i in self.path_matrix:
            if np.any(i):
                break
            zero_shift_top += 1

        for i in range(rows - 1, -1, -1):
            if np.any(self.path_matrix[i, :]):
                break
            zero_shift_bottom += 1

        for i in range(0, cols):
            if np.any(self.path_matrix[:, i]):
                break
            zero_shift_left += 1
        for i in range(cols - 1, -1, -1):
            if np.any(self.path_matrix[:, i]):
                break
            zero_shift_right += 1
        print("path 2", self.path_matrix)
        pm_without_zeros_edges = self.path_matrix[zero_shift_top + 1:-(zero_shift_bottom + 1), zero_shift_left + 1:-(zero_shift_right + 1)]
        cv2.waitKey(0)
        output = []
        for ind, val in enumerate(pm_without_zeros_edges[0]):
            if not val & self._top:
                output.append([zero_shift_top, zero_shift_left + ind])
        for ind, val in enumerate(pm_without_zeros_edges[-1]):
            if not val & self._bottom:
                output.append([rows - zero_shift_bottom - 1, zero_shift_left + ind + 1])
        for ind, val in enumerate(pm_without_zeros_edges[:, 0]):
            if not val & self._left:
                output.append([zero_shift_top + ind, zero_shift_left])
        for ind, val in enumerate(pm_without_zeros_edges[:, -1]):
            if not val & self._right:
                output.append([zero_shift_top + ind, cols - zero_shift_right - 1])
        return output

    def solveMethod(self, origImg, entryPoint):
        edges = np.copy(self.path_matrix)
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
            if sp[-1] in list(entryPoint[1]):
                break
            if edges[h][w] < 0b1111:
                rec.append(len(sp))
            if not edges[h][w] & self._top:
                # If this edges is open upwards
                edges[h][w] |= self._top
                h = h - 1
                sp.append([h, w])
                edges[h][w] |= self._bottom
                p = p + 1
                continue
            if not edges[h][w] & self._bottom:
                # If the edges is open downward
                edges[h][w] |= self._bottom
                h = h + 1
                sp.append([h, w])
                edges[h][w] |= self._top
                p = p + 1
                continue
            if not edges[h][w] & self._left:
                # If the edges is open left
                edges[h][w] |= self._left
                w = w - 1
                sp.append([h, w])
                edges[h][w] |= self._right
                p = p + 1
                continue
            if not edges[h][w] & self._right:
                # If the edges is open right
                edges[h][w] |= self._right
                w = w + 1
                sp.append([h, w])
                edges[h][w] |= self._left
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

    def pathHighlight(self, img, path):
        size = img.shape[0] / self.lines_count
        for coordinate in path:
            h = round(size * (coordinate[0] + 1))
            w = round(size * (coordinate[1] + 1))
            h0 = round(size * coordinate[0])
            w0 = round(size * coordinate[1])
            img[h0:h, w0:w] = img[h0:h, w0:w] - 50
        return img
