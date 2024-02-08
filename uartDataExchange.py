import struct
import threading
import time
import queue
from multiprocessing.connection import answer_challenge

import serial.tools.list_ports

initial_bytes = bytes([0xFA, 0xD3, 0x0F])
check_bytes = bytes([0xEF, 0xFF, 0xA0])
queue_task = queue.Queue(maxsize=3)
position_answer_bytes = bytes([0xAF, 0xE4, 0xCD])
is_UART_connected = False
lock_is_UART_connected = threading.Lock()


def get_port():
    # Find and open the COM port
    ports = serial.tools.list_ports.comports()
    if ports is not None:
        ser = None
        for port in ports:
            print(port.device)
            try:
                ser = serial.Serial(port.device, baudrate=115200, bytesize=8, parity='N',
                                    stopbits=1, timeout=1, xonxoff=False, rtscts=False, writeTimeout=1)
                ser.timeout = 0.7
                ser.write(initial_bytes)
                answer = ser.read(3)
                print("answer", answer)
                if answer == check_bytes:
                    return port, ser
                ser.close()
            except Exception as e:
                if ser is not None:
                    ser.close()
                print(e)
                continue
    return None, None


def uart_thread():
    global is_UART_connected
    while True:
        try:
            bytes_task = bytes([0, 0, 0, 0, 0, 0, 0, 0])
            port, ser = get_port()
            if port is None or ser is None:
                print("Стол не идентифицирован")
                time.sleep(2)
                continue
            with lock_is_UART_connected:
                is_UART_connected = True
            ser.timeout = 1.0
            while True:
                try:
                    # bytes_task = queue_task.get(True,1000)
                    bytes_task = bytes(struct.pack("<ff", 0x0fffffff, -90.0))
                    time.sleep(0.5)
                except queue.Empty:
                    pass
                ser.write(bytes_task)
                position_answer = ser.read(3)
                print("answer pos:", position_answer)
                if position_answer != position_answer_bytes:
                    break
        except Exception as e:
            with lock_is_UART_connected:
                is_UART_connected = False
            print("uart thread exception:", str(e))
            continue
