# -*- coding: utf-8 -*-
'''
AexServer

 @author: Daniel Sonnleithner
 @contact: daniel.sonnleithner@ini.phys.ethz.ch
 @organization: Institute of Neuroinformatics

 @copyright: (c) by Daniel Sonnleithner
 
 Created on: Aug 19, 2009
 
 @version: 0.1
 
 Main logic for AexServer
 
 Change log:
'''
from MonServer import MonServerStarter
from StimServer import StimServerStarter
from ControlServer import ControlServerStarter
import logging, threading
import os, time, stat, errno as error_number
from threading import Lock
from contextlib import contextmanager

class asyncCheckAction(threading.Thread):
    def __init__(self, poll, action, log=None, interval=1.0):
        """
        poll: function that returns False when the check fails
        action: actoin to complete when check fails
        """
        threading.Thread.__init__(self)
        self.action = action
        self.poll = poll
        self.log=log
        self.interval=interval

    def run(self):
	self.running = True
        while self.running:
            with self.do() as result:
                if not result:
                    if not self.log==None: self.log()
                    self.action()

    @contextmanager
    def do(self):
        try:
            result = self.poll()
            yield result
        finally:
            time.sleep(self.interval)
            
            
    def stop(self):
	self.running = False




class AexServer(object):
    
    deviceLock = Lock()

    def aexfs_poll(self):
        try:
            return self.aexfd_ctime == os.stat(self.aexDevice)[stat.ST_CTIME]
        except OSError as (error_number, errstr):
            if error_number==2:
                return False
            else:
                raise


    
    def __init__(self, aexDevice, monPort, stimPort, ctrlPort, prompt = True, debug = False):

        self.openAexDevice(aexDevice) 
        
        self.monServer = MonServerStarter(monPort,
                self.aexfd,
                AexServer.deviceLock,
                debug = debug,
                aexServer = self)

        self.stimServer = StimServerStarter(stimPort,
                self.aexfd,
                AexServer.deviceLock,
                debug = debug,
                aexServer = self)

        self.ctrlServer = ControlServerStarter(ctrlPort,
                self.monServer.server,
                self.stimServer.server,
                debug = debug,
                aexServer = self)

        #For automatically reconnecting when aex board is restarted
        self.checkAexDevice=asyncCheckAction(\
                self.aexfs_poll,\
                self.resetAexDevice,
                lambda: self.log("Lost device file"))
        self.open = False
        self.prompt = prompt
        self.debug = debug
        
        self.log("Aex Server started on device %s, using prompt %s, debug modus %s" % (aexDevice, prompt, debug))


    def stop(self):
        
        self.running = False
        
        self.checkAexDevice.stop()
        print("Servers stopping")
    
        self.ctrlServer.stop()
        print("Control stopped")
    
        self.monServer.stop()
        print("mon stopped")
        
        self.stimServer.stop()
        print("stim stopped")
        
        logging.shutdown()
        
        os.close(self.aexfd)
        
        
    def start(self):
        self.monServer.start()
        self.stimServer.start()
        self.ctrlServer.start()
        self.checkAexDevice.start()
        self.open=True
        self.running = True
    
        if self.prompt:
            print("Type \'stop\' to stop the server")
        
            while self.running:
                
                command = raw_input()
            
                if command == "stop":
                
                    self.running = False
                    
                elif command == "showClients":
                    
                    print("MonServerClients: \n")
                    self.monServer.server.printClients()
                    print("\nStimServerClients: \n")
                    self.stimServer.server.printClients()
                    
                elif command == "reset":
                    
                    self.resetAexDevice()
                    
            # using the ServerStarter to stop because maybe we have to clean up a virtual device
            stop_server(aexServer = self)

    
    def resetNewAexDevice(self, newDevice):
        
        self.log("resetting AEX device %s with FD: %i" % (self.aexDevice, self.aexfd))
        
        
        self.closeAexDevice()
        #Wait until aex board settles                
        self.delayRetry(newDevice, 1.0, self.log)
    
        self.log("new AEX device %s opened with FD: %i" % (self.aexDevice, self.aexfd))


    def resetAexDevice(self):
        self.resetNewAexDevice(self.aexDevice)

    def closeAexDevice(self):
        if self.open:
            self.deviceLock.acquire()
            os.close(self.aexfd)        
            time.sleep(0.1)
            self.log('Closed device file and acquired lock')
        else:
            self.log('Attempted to close closed device file')
        
    def openAexDevice(self, aexDevice=None):
        self.open = True
        if aexDevice is None:
            aexDevice = self.aexDevice
        self.aexfd =  os.open(aexDevice, os.O_RDWR | os.O_NONBLOCK)
        self.aexfd_ctime = os.stat(aexDevice)[stat.ST_CTIME]
        self.aexDevice = aexDevice
        if self.deviceLock.locked():
            self.deviceLock.release()
        self.log('Opened device file and released lock')
    
    def log(self, message):
        
        logging.warning("[AexServer: %i]: %s" % (self.aexfd, message))

    def delayRetry(self, newDevice, delay, log):
        while True:
            try:
                self.openAexDevice(newDevice) 
                self.monServer.server.updateFD(self.aexfd)
                self.stimServer.server.updateFD(self.aexfd)
                return None
            except OSError as (en, es):
                if en in (error_number.EBADF, error_number.ENOENT, error_number.EACCES):
                    log("Cannot open file, retrying in 1s")
                    time.sleep(delay)            
                    pass
                else:
                    raise

@contextmanager
def heldlock(lock):
    with lock:
        yield


virtual = False

def start_server(dev_file = 'virtual', prompt = False, monPort = 50001, stimPort = 50002, ctrlPort = 50003, debug = True):
    
    global virtual
    
    if dev_file == 'virtual':

        virtual = True
        n_virtual = 0
        while True:
            try:
                dev_file = '/tmp/aerfx2_virtual' + str(n_virtual)
                os.mkfifo(dev_file)
                break
            except OSError, e:
                if e.errno == error_number.EEXIST:
                    n_virtual += 1
                

    try:
        logging.basicConfig( filename = os.path.expanduser("~") + "/AexServer.log",
                                 format = "%(asctime)s %(levelname)s: %(message)s",
                                 datefmt = "%Y-%m-%d %H:%M:%S",
                                )
        
        aexServer = AexServer(dev_file, monPort, stimPort, ctrlPort, prompt = prompt, debug = debug)
        
        aexServer.start()
        
        return aexServer
            
    except IOError, e: 
        print(e)

        return None


def stop_server(aexServer):

    global virtual

    aexServer.running = False
    aexServer.stop()

    if virtual:
        os.remove(aexServer.aexDevice)
