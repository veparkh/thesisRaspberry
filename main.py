import threading

import bluetooth
import cv2
from uartDataExchange import uart_thread
import bluetoothDataExchange
from bluetoothDataExchange import exchange, add_crc, count_crc, is_correct_crc, coord_to_byte_arr_with_crc

server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", bluetooth.PORT_ANY))
server_sock.listen(1)
port = server_sock.getsockname()[1]
UUID = "da7ea4af-e574-4b01-aff5-e7ec710a0aeb"
bluetooth.advertise_service(server_sock, "SampleServer", service_id=UUID,
                            service_classes=[UUID, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE])
print("Waiting for connection on RFCOMM channel", port)
lock_mode = threading.Lock()
thread = threading.Thread(target=uart_thread)
thread.start()
while True:
    client_sock, client_info = server_sock.accept()  # Цикл на подключение разных устройств
    print("Accepted connection from", client_info)
    exchange(client_sock, lock_mode)
    print("after exchange")

