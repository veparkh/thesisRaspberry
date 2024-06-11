import threading
import bluetooth
from uartDataExchange import uart_thread
from bluetoothDataExchange import exchange


# Настройка bluetooth
server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", bluetooth.PORT_ANY))
server_sock.listen(1)
port = server_sock.getsockname()[1]
UUID = "da7ea4af-e574-4b01-aff5-e7ec710a0aeb"
bluetooth.advertise_service(server_sock, "SampleServer", service_id=UUID,
                            service_classes=[UUID, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE])
print("Waiting for connection on RFCOMM channel", port)
# Создание блокировки для доступа к глобальной переменной mode
lock_mode = threading.Lock()
# Создание и запуск потока под uart
thread = threading.Thread(target=uart_thread)
thread.start()

while True:
    client_sock, client_info = server_sock.accept()  # Цикл на подключение разных устройств
    print("Accepted connection from", client_info)
    exchange(client_sock, lock_mode)
    print("after exchange")

