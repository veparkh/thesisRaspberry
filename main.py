import bluetooth
import cv2
import numpy as np

server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", bluetooth.PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]

UUID = "da7ea4af-e574-4b01-aff5-e7ec710a0aeb"

bluetooth.advertise_service(server_sock, "SampleServer", service_id=UUID,
                            service_classes=[UUID, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE],
                            # protocols=[bluetooth.OBEX_UUID]
                            )

print("Waiting for connection on RFCOMM channel", port)

client_sock, client_info = server_sock.accept()
print("Accepted connection from", client_info)

try:
    while True:
        message = input("Enter message: ")
        client_sock.send(message.encode("utf-8"))
except OSError:
    pass

print("Disconnected")

client_sock.close()
server_sock.close()
print("All done.")





