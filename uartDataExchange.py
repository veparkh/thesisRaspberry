import threading
import time
import queue
import serial.tools.list_ports

initial_bytes = bytes([0xFA,0xD3,0x0F])
check_bytes = bytes([0xEF,0xFF,0xA0])
queue_task = queue.Queue(maxsize=3)
is_UART_connected = False
lock_is_UART_connected = threading.Lock()


def uart_thread():
    global is_UART_connected
    while True:
        try:
            # Find and open the COM port
            ports = serial.tools.list_ports.comports()
            port = None
            ser = None
            if ports is None:
                raise ValueError("No COM port found.")
            for i in ports:
                print(type(i.device))

                ser = serial.Serial(i.device, baudrate=115200, bytesize=8, parity='N',
                                    stopbits=1, timeout=1, xonxoff=False,rtscts=False,writeTimeout=1)
                print(f"{ser}")
                ser.timeout = 0.7
                try:
                    ser.write(initial_bytes)
                except serial.SerialTimeoutException as e:
                    print(e)
                    continue
                answer = ser.read(3)
                print("herr4e")
                if answer != check_bytes:
                    print("wrong port")
                    ser.close()
                    continue
                port = i
                break
            if port is None:
                print("Стол не идентифицирован")
                time.sleep(4)
                continue
            with lock_is_UART_connected:
                is_UART_connected = True
            while True:
                bytes_task = queue_task.get()
                ser.write(bytes_task)
        # except ValueError as ve:
        #     print("Error:", str(ve))

        except serial.SerialException as se:
            print("Serial port error:", str(se))
            continue

        # except Exception as e:
        #     print("An error occurred:", str(e))