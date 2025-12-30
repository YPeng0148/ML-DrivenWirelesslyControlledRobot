import socket
import serial

uart = serial.Serial('/dev/ttyTHS1', baudrate=115200)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 8888))

print("Jetson is listening on port 8888")

while True:
    data, addr = sock.recvfrom(1024)
    command = data.decode('utf').strip()
    print(command)
    uart.write((command  + '\n').encode('utf-8'))
