import socket
import random
import time
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host ="127.0.0.1"
port =8000
s.connect((host,port))

def ts(send_data):
   s.send(send_data.encode()) 
   data = ''
   data = s.recv(1024).decode()
   print (data)

def main(argv):
  n = 5
  t = 1
  if(len(argv) == 2):
    n = int(argv[0])
    t = int(argv[1])

  it = 0

  while (1 and it < n):
     r = str(random.randint(-100,100)) +','+str(random.randint(-100,100)) + ',' + str(random.randint(-100,100)) + ',' + str(t)
     ts(r)
     time.sleep(t)
     it += 1

  s.close()

if __name__ == "__main__":
  main(sys.argv[1:])