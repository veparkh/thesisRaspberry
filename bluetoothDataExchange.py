from datetime import datetime
from maze_solving import Maze
import struct
import threading
import time
import cv2
import zlib
import queue

import maze_solving
from bluetooth import BluetoothSocket
from dataclasses import dataclass
import uartDataExchange


@dataclass
class Angles:
    roll: float
    pitch: float


queue_angles = queue.Queue()

autoControlCommand = bytearray([255, 1])
manualControlCommand = bytearray([255, 2])
mapBuildingCommand = bytearray([255, 3])
chooseModeCommand = bytearray([255, 4])
imageCommand = bytearray([100, 3])
coordCommand = bytearray([100, 2])

mode = 0  # 0 - нет соединения/не выбран 1 - автоконтроль 2- ручное управление 3 - построение карты 4 - выбор режима
# -1 - соединение разорвано
coord = [0.0, 0.0]
image = False


def byte_array_to_angles(arr: bytearray):
    angle1 = struct.unpack('>f', arr[0:4])
    angle2 = struct.unpack('>f', arr[4:8])
    return Angles(angle1, angle2)


def coord_to_byte_arr_with_crc(x: float, y: float):
    arr = bytearray(struct.pack(">ff", x, y))
    return add_crc(coordCommand + arr)


def is_correct_crc(arr: bytearray):
    crc = int.from_bytes(arr[-4:], 'big')
    crc2 = count_crc(arr[0:-4])
    return crc == crc2


def count_crc(arr: bytearray):
    return zlib.crc32(arr)


def add_crc(arr: bytearray):
    crc = zlib.crc32(arr)
    crc_bytes = crc.to_bytes(4, 'big')
    return arr + crc.to_bytes(4, 'big')


def image_to_byte_array(img):
    img2 = cv2.resize(img, (300, 700), 1, 1, cv2.INTER_AREA)
    result, byteArray = cv2.imencode(".png", img2)
    print("array size", len(byteArray))
    if result:
        return byteArray
    else:
        return None


def handle_incoming_message(data, lock: threading.Lock):
    global mode
    global image
    if data[0] == 255 and data[1] in range(1, 5):
        with lock:
            mode = data[1]

    if data[0] == 100 and data[1] in range(1, 5):
        if data[1] == 1:
            angles = byte_array_to_angles(data[2:])
            # print("angles:", angles.roll, angles.pitch)

            print("time", datetime.now(), "     ", angles)
            queue_angles.put(angles)
        elif data[1] == 2:
            pass
        elif data[1] == 3:
            image = True
        elif data[1] == 4:
            pass


def b_recv_messages_thread_f(socket: BluetoothSocket, lock: threading.Lock):
    datetime.utcnow().strftime('%F %T.%f')[:-3]
    print(type(socket))
    global mode
    try:
        while True:
            data = socket.recv(20)
            print("длина пакета", len(data))
            if data is None or len(data) == 0:
                print("data is None")
                with lock:
                    mode = -1
                socket.close()
                break
            handle_incoming_message(data, lock)
    except OSError:
        print("exception")
        with lock:
            mode = -1


def auto_control(client_sock: BluetoothSocket, lock):
    client_sock.send(add_crc(autoControlCommand))
    maze = Maze()
    cap = cv2.VideoCapture(1)
    prev_x_err = 0
    prev_y_err = 0
    out_point = None
    x_err_sum = 0
    y_err_sum = 0
    exit_path = None
    weight_matrix = None
    ret, frame = cap.read()
    prev_coord = [0, 0]
    global image
    while True:
        with lock:
            print(f"mode {mode}")
            if mode != 1:
                with uartDataExchange.lock_is_UART_connected:
                    if uartDataExchange.is_UART_connected:
                        uartDataExchange.queue_task.put(Angles(0, 0))
                image = True
                break
        time.sleep(0.01)
        ret, frame = cap.read()
        if not ret:
            continue
        start = time.time()
        x_goal, y_goal,x_curr,y_curr, img = maze.solve_maze(frame,False)

        print("x_goal x_curr, y_goal, y_curr",x_goal,x_curr, y_goal,y_curr)
        if image:
            img_maze = maze.draw_path_matrix()

            if img_maze is None:
                continue
            img_grid = maze.draw_grid(img_maze)
            cv2.imshow("img_grid",img_grid)
            byte_array = image_to_byte_array(img_maze)
            client_sock.send(add_crc(imageCommand + byte_array.size.to_bytes(4, 'big')))
            client_sock.send(byte_array)
            image = False
        if x_goal is None:
            client_sock.send(coord_to_byte_arr_with_crc(400 / maze.aligned_img_len, 400 / maze.aligned_img_len))
            print("Координаты шарика не определены")
            continue
        client_sock.send(coord_to_byte_arr_with_crc(x_curr / maze.aligned_img_len, y_curr / maze.aligned_img_len))
        print(x_goal,x_curr,y_goal,y_curr)
        x_error = x_goal - x_curr
        y_error = y_goal - y_curr
        dt = round((time.time() - start) * 1000)
        x_angle = x_error * 0.14  # + 0.02 * dt*maze.block_count * (prev_x_err-x_error) + 0.0001 * x_err_sum
        print("x_angle", x_angle)
        y_angle = y_error * 0.14  # + 0.02 * dt * maze.block_count * (prev_y_err - y_error) + 0.0001 * y_err_sum
        print("y_angle", y_angle)
        prev_x_err = x_error
        prev_y_err = y_error
        x_err_sum += x_error
        y_err_sum += y_error
        cv2.waitKey(1)
        if abs(x_err_sum) > 400:
            x_err_sum = 0
        if abs(y_err_sum) > 400:
            y_err_sum = 0
        with uartDataExchange.lock_is_UART_connected:
            if uartDataExchange.is_UART_connected:
                uartDataExchange.queue_task.put(Angles(roll=-x_angle, pitch=y_angle))


def manual_control(socket: BluetoothSocket, lock):
    socket.send(add_crc(manualControlCommand))
    print(add_crc(manualControlCommand))
    global mode
    while True:
        with lock:
            if mode != 2:
                with uartDataExchange.lock_is_UART_connected:
                    if uartDataExchange.is_UART_connected:
                        uartDataExchange.queue_task.put(Angles(0, 0))
                break
        try:
            angles = queue_angles.get(True, 1)
            with uartDataExchange.lock_is_UART_connected:
                if uartDataExchange.is_UART_connected:
                    uartDataExchange.queue_task.put(Angles(angles.pitch[0], angles.roll[0]))
        except queue.Empty:
            print("manual control exception")
            pass


def exchange(socket: BluetoothSocket, lock: threading.Lock):
    print("locked", lock.locked())
    thread = threading.Thread(target=b_recv_messages_thread_f, args=(socket, lock))
    thread.start()
    global image
    global mode
    is_first_waiting = True
    try:
        while True:
            with lock:
                mode_local = mode
            if mode_local != 4:
                is_first_waiting = True
            if mode_local == 0:
                time.sleep(0.02)
                continue
            elif mode_local == -1:
                thread.join()
                print("thread finished")
                socket.close()
                print("socket closed")
                with lock:
                    mode = 0
                break
            elif mode_local == 1:
                image = True
                auto_control(socket, lock)
            elif mode_local == 2:
                manual_control(socket, lock)
                pass
            elif mode_local == 3:
                pass
            elif mode_local == 4:
                if is_first_waiting:
                    socket.send(add_crc(chooseModeCommand))
                    print("chooseModeCommand send")
                    is_first_waiting = False
                    time.sleep(0.02)
                pass
    except OSError:
        print("send exception")
        thread.join()
        print("thread finished")
        socket.close()
        print("Disconnected")
