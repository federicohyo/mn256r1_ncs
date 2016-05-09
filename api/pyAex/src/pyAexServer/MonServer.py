'''
AexServer -- MonServer

 @author: Daniel Sonnleithner
 @contact: daniel.sonnleithner@ini.phys.ethz.ch
 @organization: Institute of Neuroinformatics

 @copyright: (c) by Daniel Sonnleithner
 
 Created on: Aug 19, 2009
 
 @version: 0.2
 
 Provides access for many clients to the output of AEX boards
 
 Change log:
 
 11-06-06: filtering of events per client controlled by ControlServer
 
'''
from SocketServer import ThreadingTCPServer
from SocketServer import BaseRequestHandler
from threading import Thread, Lock
import time
import logging
import errno
import os

import pylab

import numpy as np

TIMERESCALE = 6.25 # 50 MHz oscillator on AEXL board, 8 samples, 50/8 = 6.25.

class MonServer(ThreadingTCPServer,Thread):
    '''
    classdocs
    '''
    clientsLock = Lock()

    def __init__(self, port, deviceFd, deviceLock, debug = False, aexServer = None):
        '''
        Constructor
        '''
        self.running = False
        
        self.__clients = {}
        self.debug = debug
        self.aexServer=aexServer
        
        self.__deviceFd = deviceFd
        self.deviceLock=deviceLock
        
        ThreadingTCPServer.__init__(self, ("", port), MonNetCom, False)
        
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
        with MonServer.clientsLock:
            self.__clients[client.client_address] = client
                
        return True
    
    
    def disconnect(self, client):
        '''
        When a client is disconnected from the server, the NetControlCom object calls this method. 
        
        The client is removed from the client database (self.__clients list)
        '''
        with MonServer.clientsLock:
            del self.__clients[client.client_address]
        
        
    def run(self):
        
        self.log("Mon server starts")
        
        self.running = True
        
        while(self.running):
            wait=False
            with self.deviceLock:
                try:
                    receivedData = os.read(self.__deviceFd, 16384)
                    
                    if not receivedData == '':
                        receivedData_array = np.fromstring(receivedData, dtype='uint32').reshape(-1,2)
                        # in AEX 1 is 1000/128 us that is about 8
                        receivedData_array[:,0] = receivedData_array[:, 0]/TIMERESCALE 
                        if self.debug: self.log("Broadcasting "+str(receivedData_array[0,0]))
                        
                        self.broadcastData(receivedData_array)
                    
                except OSError, tryAgain:

                    if not tryAgain.errno == errno.EAGAIN:
                        raise tryAgain 
                    else:
                        wait=True
                           
                except Exception, e:
                    self.log("Exception during monitoring: %s" % str(e))
            if wait:
                time.sleep(0.001)                
                
    
    def stop(self):

        self.running = False
        
        time.sleep(0.5)    
    
        self.shutdown()
        
        self.log("Mon server shuts down")   
    
    
    def updateFD(self, deviceFD):
        '''
        updates the current device file descriptor
        
        ATTENTION: never call this function without having acquired the deviceLock!
        '''
        self.__deviceFd = deviceFD
            
    
    def broadcastData(self, data):
        
        with MonServer.clientsLock:
        
            for client in self.__clients.values():
                
                client.sendData(np.copy(data))
            
    
    
    def broadcast(self, message):
        
        with MonServer.clientsLock:
        
            for client in self.__clients:
                
                client.send(message)
        
    
    def addMask(self, client, mask, checkValue):
        
        with MonServer.clientsLock:
        
            self.__clients[client].addMask(mask, checkValue)
        
        
    def clearMasks(self, client):
        
        with MonServer.clientsLock:
        
            self.__clients[client].clearMasks()
    
    
    def printClients(self):
        
        with MonServer.clientsLock.acquire():
            
            for client in self.__clients.keys():
                
                print(str(client))
        
    
    def log(self, message):
        
        logging.warning("[MonServer: %i]: %s" % (self.__deviceFd, message))
        


class MonServerStarter(Thread):
    
    def __init__(self, port, deviceFd, deviceLock, debug = False, aexServer = None):
        
        self.server = MonServer(port, deviceFd, deviceLock, debug = debug, aexServer = aexServer)
        
        Thread.__init__(self)
        
        
    def run(self):
        
        self.server.setDaemon(True)
        
        self.server.start()
        
        self.server.serve_forever()
        
        
    def stop(self):
        
        self.server.stop()
        
        

class MonNetCom(BaseRequestHandler):
    '''
    each time a client connects to the server, a new MonNetCom object is created that registers at the
    server
    '''
    masklock = Lock()        

    def addMask(self, mask, checkValue):
        '''
        add mask and checkValue -> see sendData
        '''
        with MonNetCom.masklock:
            self.masks.append((mask, checkValue))
        
        
    
    def clearMasks(self):
        '''
        clears all masks and checkValues -> see sendData
        '''
        with MonNetCom.masklock:
            self.masks = []

    
    def handle(self):
        '''
        handles all incoming traffic. In the case of the MonServer it is more or less a placebo
        method because nothing useful is done with incoming data. 
        '''
        if self.isConnected:
        
            while self.isConnected:
            
                try:
                    data = self.request.recv(8)
                    
                    print(repr(data))
                    
                    if not data:
                        
                        self.isConnected = False
                
                except IOError, ioe:
                    
                    if ioe.errno == errno.ECONNRESET:
                        
                        self.server.log("connection reset by peer - better wait a bit after last stimulation")
                    else:
                        raise ioe
                    
        else:
           
            self.request.close()
                
    
    def setup(self):
        '''
        when creating a connection, this method is called. It registers this object at the server
        '''
        self.isConnected = self.server.connect(self)
                
        self.masks = []
    
    def finish(self):
        '''
        when closing the connection, this method is called. It tells the server to disconnect
        '''
        self.server.disconnect(self)
        
    
    def sendData(self, data):
        '''
        sends data (nparrays) to clients
        
        provides the possibility to filter the addresses with given mask(s) and a checkValue(s)
        first the mask is applied and then checked if it is identical to the checkValue; if so
        these events are forwarded; all events where the addresses don't match one of the given 
        filter rules are not forwarded.
        '''
        if len(self.masks) > 0:
            with MonNetCom.masklock:
      
                s = set()

                for mask in self.masks:                                    
                    g = data[:,1] & mask[0]        
                    s.update(pylab.find(g == mask[1]))
                
                idx=list(s)
                idx.sort()
                data = data[idx]
            
        
        self.send(data.flatten().tostring())
        
    
    def send(self, message):
        '''
        sends message (strings) to the client
        '''
        try:
            self.request.sendall(message)
        
        except IOError, ioe:
            if ioe.errno == errno.EPIPE:
                
                self.server.log("broken pipe: %s" % ioe)
            else:
                raise ioe
