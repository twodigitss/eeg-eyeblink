"""
    RECURSOS
        https://forums.raspberrypi.com/viewtopic.php?t=329762
        https://raspberrypi.stackexchange.com/questions/24028/using-python-with-bluetooth-to-communicate
        https://blog.adafruit.com/2020/04/17/starting-with-raspberry-pi-bluetooth-python-python-bluetooth-raspberrypi-piday-raspberry_pi/
        https://blog.adafruit.com/2018/05/28/recording-brainwaves-with-a-raspberry-pi/

        https://pythonic.rapellys.biz/articles/interfacing-with-bluetooth-devices-using-python/

    PASOS QUE HACE: -> 
        1) los datos del muse vienen aqui en formato CSV
        2) leer los datos del stream 
            suponiendo que es una conexion directa y 
            que los datos llegan del stream, deberia canalizar la senal
            y recuperar los datos del bluetooth mismo segun llegan (csv??)

"""

# https://pybluez.readthedocs.io/en/latest/
# https://pybluez.readthedocs.io/en/latest/api/bluetooth_socket.html
# NOTE: wow, no words...
import bluetooth
import time
devices = {}
socket: bluetooth;

def scann():
    global devices
    print("Scanning... (5 seconds)")
    done_finding = False

    while done_finding is False:
        nearby_devices = bluetooth.discover_devices(
                duration=5, flush_cache=True)

        # print the name and MAC address of each device
        for addr in nearby_devices:
            devices[bluetooth.lookup_name(addr)] = addr
            # print("Device Name:", bluetooth.lookup_name(addr))
            # print("Device MAC Address:", addr)
        if bool(devices):
            done_finding = True
        print("Nothing found, trying again...")
    
    print(f"Found: {devices}!")

# establish a connection with the device
# TODO: does the connection keeps up after finishing the method???
def connect():
    global devices, socket
    connected = False
    address = devices['Pixel 7a']
    port = 1; connected = False

    while not connected:
        print(f"Trying to connect in port {port} ...")
        time.sleep(5)
        try:
            # ES IMPORTANTE CREAR UN SOCKET NUEVO, PERDI 4 HORAS EN ESTO.
            # GRACIAS A https://github.com/pybluez/pybluez/issues/191
            socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            socket.connect((address, port))
            # socket.accept()
            print(f"connected to port {port}!")
            connected = True
            port = 1
        except Exception as e:
            print(f"Error: {e}")
            port = port + 1
    
    data_stream()



# TODO: UNTESTED
def data_stream():
    global socket
    print("5 seconds cooldown between each send/recieve")

    while True:
        time.sleep(5)
        try:
            message = "Hello, world!"

            print("Sending message...")
            socket.send(message)

            print("Receiving message...")
            message = socket.recv(1024)
            print("!!!Received message:", message)

        except bluetooth.btcommon.BluetoothError as e:
            print("Error 2nd nest:", e)


if __name__ == "__main__":
    scann()
    connect()

    # asd = input("type enter when you want to proceed. ")
    #
    # data_stream()
    socket.close()
    print("Closing socket, Bonjourno!")





# EXAMPLE 1 = i also thought about for running raw commands
# from subprocess import run
# run("command", shell=True, capture_output=True, text=True).stdout.strip()

#EXAMPLE 2
# import asyncio
# import bleak
# from bleak import BleakClient
#
# address = "CC:C8:41:10:2C:3B" # Sensor MAC
# MODEL_NBR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e" # Sensor TX Characteristic
#
# async def main(address):
#     client = bleak.backends.bluezdbus.client.BleakClientBlueZDBus(address)
#     try:
#         await client.connect()
#
#         #paired = await client.pair(protection_level=2)
#         #print(f"Paired: {paired}")
#
#         model_number = await client.read_gatt_char(MODEL_NBR_UUID)
#         print("Data: {0}".format("".join(map(chr, model_number))))
#     except Exception as e:
#         print(e)
#     finally:
#         #await client.disconnect()
#         print("Done")
#
# asyncio.run(main(address))

