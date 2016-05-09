# -*- coding: utf-8 -*-
'''
AexServer -- StimServer

 @author: Daniel Sonnleithner
 @contact: daniel.sonnleithner@ini.phys.ethz.ch
 @organization: Institute of Neuroinformatics

 @copyright: (c) by Daniel Sonnleithner
 
 Created on: Aug 20, 2009
 
 @version: 0.1
 
 Several clients can stimulate AEX boards at the same time using that server
 
 Change log:
'''
from Queue import Queue
from SocketServer import ThreadingTCPServer
from SocketServer import BaseRequestHandler
from threading import Thread, Lock, Event
import time
import logging
import errno
import os

#For FIONREAD (reading the number of bytes without reading data from socket buffer
import fcntl
import termios

import numpy as np

TIMERESCALE = 8.0 # 125 MHz oscillator, 1000./125.

class StimServer(ThreadingTCPServer,Thread):
    '''
    classdocs
    '''
    clientsLock = Lock()
    
    queueHasNewElement = Event()

    def __init__(self, port, deviceFd, deviceLock, debug = False, aexServer = None):
        '''
        Constructor
        '''
        self.running = False
        
        self.__dataQueue = Queue(0)
        
        self.__clients = []
        self.debug = debug
        
        self.__deviceFd = deviceFd
        self.deviceLock = deviceLock
        self.aexServer=aexServer
        
        self.runningTime = 0
        self.receivingTime = 0
        self.packetsReceived = 0
        self.sleepingTimes = 0
        
        ThreadingTCPServer.__init__(self, ("", port), StimNetCom, False)
        
        ThreadingTCPServer.allow_reuse_address = True
        
        self.server_bind()
        self.server_activate()
        
        Thread.__init__(self)
        
        
    def connect(self, client):
        '''
        When a new client is connecting to the server, a new NetControlCom object is created. 
        This NetControlCom object then registers at the server with this connect method. 
        
        The client is put into the client database (self.__clients list). 
        '''        
        StimServer.clientsLock.acquire()
        
        self.__clients.append(client)
                
        StimServer.clientsLock.release()
                
        return True
    
    
    def disconnect(self, client):
        '''
        When a client is disconnected from the server, the NetControlCom object calls this method. 
        
        The client is removed from the client database (self.__clients list)
        '''
        StimServer.clientsLock.acquire()
               
        self.__clients.remove(client)
        
        StimServer.clientsLock.release()
        
        
    def putData(self, data):
        '''
        Put a data into the data queue
        '''
        self.__dataQueue.put(data)
        
        StimServer.queueHasNewElement.set()
        
    
    def run(self):
        if self.debug:
            trapc=0
            trapc2=0
        
        self.log("Stim server starts -- Debug mode is %s" % self.debug)
        self.reset=False        
        
        self.running = True
        
        while self.running:
            
            if self.__dataQueue.empty():
                
                StimServer.queueHasNewElement.wait()
                
                StimServer.queueHasNewElement.clear()
                
            else:
                try:                    
                    data = self.__dataQueue.get()                    
                    data_array = np.fromstring(data, dtype='uint32').reshape(-1,2)                    
                    
                    if data_array[0,0]==np.array(-2,dtype='uint32'): #-1 in unit32  
                        self.log('Received hold and reset event')                                      
                        self.reset=True                                                              
                        self.aexServer.closeAexDevice()  
                        if len(data_array)==1: #Reset event has been sent alone
                            continue
                        else:                            
                            data_array=data_array[1:,:]   
                                                                                                                    
                    # in AEX 1 is 1000/128 us that is 7.8125, slightly smaller is better than bigger, otherwise rounding errors will lead to strong delays

                    
                    if data_array[0,0]==np.array(-1,dtype='uint32'): #-1 in unit32
                        self.log('Received go event')                                                                                                                        
                        data_array=data_array[1:,:]                        
                        data = data_array.tostring()
                    else:
                        data_array[:,0] = np.round(data_array[:, 0] * TIMERESCALE).astype('uint32')
                        data = data_array.tostring()
                    
                    sent = False
                    
                    while not sent:
                        wait = False                        
                        if self.reset:
                            self.aexServer.openAexDevice()                                                        
                            self.reset=False 
                                              
                        with self.deviceLock:                            
                            try:                                                                           
                                c = os.write(self.__deviceFd, data)                                
                                if c == len(data):
                                    sent = True
                                else:
                                    data = data[c:]
                            
                            except OSError, ose:
                                if not ose.errno == errno.EAGAIN:
                                    raise ose
                                else:
                                    wait=True
                            finally:
                                pass
                        #end deviceLock context 
                        if wait: 
                            time.sleep(0.001)
                                
                                
                except Exception, e:
                        
                        self.log("Exception during sending data: %s" % str(e))
                        
#                        if self.deviceLock.locked():
#                            self.deviceLock.release()

  
    def stop(self):
        
        self.running = False
        
        time.sleep(0.5)
        
        self.shutdown()
        
        self.log("Stim server shuts down")
     
     
    def updateFD(self, deviceFD):
        '''
        updates the current device file descriptor
        
        ATTENTION: never call this function without having acquired the deviceLock!
        '''
        self.__deviceFd = deviceFD   
     
        
    def printClients(self):
        
        StimServer.clientsLock.acquire()
        
        for client in self.__clients:
            
            print(str(client.client_address))
            
        StimServer.clientsLock.release()
        
    
    def log(self, message):
        
        logging.warning("[StimServer: %i]: %s" % (self.__deviceFd, message))
        
        
        
class StimServerStarter(Thread):
    
    def __init__(self, port, deviceFd, deviceLock, debug = False, aexServer = None):
        
        self.server = StimServer(port, deviceFd, deviceLock, debug = debug, aexServer = aexServer)
        
        Thread.__init__(self)
        
        
    def run(self):
        
        self.server.setDaemon(True)
        
        self.server.start()
        
        self.server.serve_forever()
        
        
    def stop(self):
        
        self.server.stop()
        
        

class StimNetCom(BaseRequestHandler):
    '''
    each time a client connects to the server, a new StimNetCom object is created that 
    registers at the server
    '''

    def handle(self):
        '''
        handles all incoming data
        '''
        if self.isConnected:
        
            while self.isConnected:
                
                self.request.recv(0)
                buf_n=np.fromstring(fcntl.ioctl(self.request,termios.FIONREAD,"xxxx"),'uint32')
                recv_n=min(8*(buf_n/8),4096)
                if recv_n>0:
                    data = self.request.recv(recv_n)
                    if not data:                    
                        self.isConnected = False
                    else:
                        self.server.putData(data)
                if buf_n==0:
                    self.isConnected = False

        else:
           
            self.request.close()
                
    
    def setup(self):
        '''
        when creating a connection, this method is called. It registers this object at the server
        '''
        self.isConnected = self.server.connect(self)
                
    
    def finish(self):
        '''
        when closing the connection, this method is called. It tells the server to disconnect
        '''
        self.server.disconnect(self)
