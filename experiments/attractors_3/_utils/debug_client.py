import socket
import sys
import struct
import numpy as np
from termios import FIONREAD
import fcntl
import time

debug=True
fps=1./40

HOST=str(sys.argv[1])
PORT=int(sys.argv[2])

if debug:
    print HOST
    print PORT


# Define vectorized.
vhex = np.vectorize(hex)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect((HOST, PORT))
sock.settimeout(fps)

while True:
    t0=time.time()
    buf_n=np.fromstring(fcntl.ioctl(sock,FIONREAD,"xxxx"),'uint32')
    recv_n=8*(buf_n/8)
    #Do the actual recieve
    if not buf_n<8:
        eventsPacket=sock.recv(recv_n)
        if sys.byteorder=='big':
            print
            vhex(np.fromstring(eventsPacket,dtype='uint32').byteswap().reshape(-1,2))
        else:
            #print vhex(np.fromstring(eventsPacket,dtype='uint32').reshape(-1,2))
            print np.fromstring(eventsPacket,dtype='uint32').reshape(-1,2)

    #Fastest frame rate should be 1/fps. Wait in case we've been faster
    t_left=fps-(time.time()-t0)
    if t_left>0: time.sleep(t_left)


