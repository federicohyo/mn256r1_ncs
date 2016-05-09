import socket
import sys
import struct
import numpy as np
from termios import FIONREAD
import fcntl
import time
import threading
 

#used to run the monitoring in background
class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        self._target = target
        self._args = args
        self._return = None
        threading.Thread.__init__(self)
 
    def run(self):
        self._target(*self._args)

    def join(self):
        threading.Thread.join(self)
        return self._return

#return monitored activity
def tcp_read(host,port,duration, fps=1./40):
    '''
    reads event from server
    '''

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.connect((host, port))
    sock.settimeout(fps)

    start = time.time()
    time.clock()    
    elapsed = 0
    spike_read = []
    while elapsed < duration:
        elapsed = time.time() - start
        #print "loop cycle time: %f, seconds count: %02d" % (time.clock() , elapsed) 
        #time.sleep(1)
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
                print np.fromstring(eventsPacket,dtype='uint32')
                spike_read.append(np.fromstring(eventsPacket,dtype='uint32'))
                #spike_read.append()#.reshape(-1,2))

        #Fastest frame rate should be 1/fps. Wait in case we've been faster
        t_left=fps-(time.time()-t0)
        if t_left>0: time.sleep(t_left)


    spike_read = np.array(spike_read)
    return spike_read


