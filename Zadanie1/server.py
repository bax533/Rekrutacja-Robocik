import socket
import sys

preva1, preva2, preva3 = 0,0,0

def calculate_v(a1, a2, a3, t):
  return (abs(a1 - preva1) + abs(a2 - preva2) + abs(a3 - preva3)) / t


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "127.0.0.1"
port = 8000
print (host)
print (port)
serversocket.bind((host, port))

serversocket.listen(5)
print ('server started and listening')
(clientsocket, address) = serversocket.accept()
print ("connection found!")
while 1:
    data = clientsocket.recv(1024).decode()
    points = data.split(',')
    intPoints = [int(points[0]), int(points[1]), int(points[2]), int(points[3])]
    print (calculate_v(intPoints[0], intPoints[1], intPoints[2], intPoints[3]))
    preva1, preva2, preva3 = intPoints[0], intPoints[1], intPoints[2]

    r='Receieve'
    clientsocket.send(r.encode())
