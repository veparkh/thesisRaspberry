import cv2
import numpy as np
import bluetooth


def image_to_byte_array(img):
    img2 = cv2.resize(img, (300, 700), 1, 1, cv2.INTER_AREA)
    cv2.imshow("changed", img2)
    cv2.waitKey(0)
    result, byteArray = cv2.imencode(".png", img2)
    print("array size", len(byteArray))
    if (result):
        return byteArray
    else:
        return None


def exchange():
    img = cv2.imread("OPEN.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("org", img)
    byteArray = image_to_byte_array(img)
    print("len", len(byteArray))
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
            client_sock.send(byteArray)
    except OSError:
        pass

    print("Disconnected")

    client_sock.close()
    server_sock.close()
    print("All done.")
