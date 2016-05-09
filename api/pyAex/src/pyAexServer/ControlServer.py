'''
AexServer -- ControlServer

 @author: Daniel Sonnleithner
 @contact: daniel.sonnleithner@ini.phys.ethz.ch
 @organization: Institute of Neuroinformatics

 @copyright: (c) by Daniel Sonnleithner
 
 Created on: Jun 1, 2011
 
 @version: 0.1
 
 Provides a possibility to give control commands to the AexServer
 
 Change log:
'''
from Queue import Queue
from SocketServer import ThreadingTCPServer
from SocketServer import BaseRequestHandler
from threading import Thread, Lock, Event
import time
import logging
import errno

class ControlServer(ThreadingTCPServer,Thread):
    '''
    classdocs
    '''
    clientsLock = Lock()
    
    queueHasNewElement = Event()
    
    def __init__(self, port, monServer, stimServer, debug = False, aexServer = None):
        '''
        Constructor
        '''
        self.running = False
        
        self.__commandQueue = Queue(0)
        
        self.__clients = []
        self.debug = debug
        
        self.aexServer = aexServer
        self.monServer = monServer
        self.stimServer = stimServer
        
        ThreadingTCPServer.__init__(self, ("", port), CtrlNetCom, False)
        
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
        ControlServer.clientsLock.acquire()
        
        self.__clients.append(client)
                
        ControlServer.clientsLock.release()
                
        return True
    
    
    def disconnect(self, client):
        '''
        When a client is disconnected from the server, the NetControlCom object calls this method. 
        
        The client is removed from the client database (self.__clients list)
        '''
        ControlServer.clientsLock.acquire()
               
        self.__clients.remove(client)
        
        ControlServer.clientsLock.release()
        
        
    def run(self):
        
        self.log("Control server starts - debug mode %s" % self.debug)
        
        self.running = True
        
        while(self.running):
        
            if self.__commandQueue.empty():
                
                ControlServer.queueHasNewElement.wait()
                
                ControlServer.queueHasNewElement.clear()
                
            else:
                
                command = self.__commandQueue.get()
                
                result = self.parseCommand(command)
                
    
    def stop(self):

        self.running = False
        
        ControlServer.queueHasNewElement.set()
        
        self.monServer = None
        self.stimServer = None
        
        time.sleep(0.5)    
    
        self.shutdown()
        
        self.log("Control server shuts down")   
    
    
    def putCommands(self, commands):
        '''
        Put a command into the command queue
        '''
        for i in range(0,len(commands) - 1):
            self.__commandQueue.put(commands[i])
        
        ControlServer.queueHasNewElement.set()
        
        
    def parseCommand(self, command):
        '''
        parse commands
        '''
        try:
            
            argument = eval(command.partition('(')[1] + command.partition('(')[2])
            
            command = command.partition('(')[0]
            
            if command == "addMask":
                
                if self.debug:
                    
                    print("addMask with argument %s" % str(argument))
                
                return self.executeAddMask(argument)
            
            elif command == "clearMasks":
                
                if self.debug:
                    
                    print("clearMasks with argument %s" % str(argument))
                
                return self.executeClearMasks(argument)
            
            else:
                
                if self.debug:
                    
                    print("unknown command: %s with argument %s" % (command, str(argument)))
                
                return "unknown command;" 
            
        except Exception, e:
            
            self.log("exception: %s" % e)
            
            return "error while parsing;"
        
        
    def executeAddMask(self, argument):
        
        self.monServer.addMask(argument[0], argument[1], argument[2])
        
        return "ok;"
        
    
    def executeClearMasks(self, argument):
        
        self.monServer.clearMasks(argument)
        
        return "ok;"
        
    
    def printClients(self):
        
        ControlServer.clientsLock.acquire()
        
        for client in self.__clients:
            
            print(str(client.client_address))
            
        ControlServer.clientsLock.release()
        
    
    def log(self, message):
        
        logging.warning("[ControlServer]: %s" % (message))
        


class ControlServerStarter(Thread):
    
    def __init__(self, port, monServer, stimServer, debug = False, aexServer = None):
        
        self.server = ControlServer(port, monServer, stimServer, debug = debug, aexServer = aexServer)
        
        Thread.__init__(self)
        
        
    def run(self):
        
        self.server.setDaemon(True)
        
        self.server.start()
        
        self.server.serve_forever()
        
        
    def stop(self):
        
        self.server.stop()
        
        

class CtrlNetCom(BaseRequestHandler):
    '''
    each time a client connects to the server, a new MonNetCom object is created that registers at the
    server
    '''

    def handle(self):
        '''
        handles all incoming traffic
        '''
        if self.isConnected:
        
            while self.isConnected:
            
                message = self.request.recv(1024)
                
                if not message:
                    
                    self.isConnected = False
                    
                else:
                    
                    self.messageBuffer = self.messageBuffer + message
                    
                    commands = self.messageBuffer.split(";")
                    
                    self.messageBuffer = commands[len(commands) - 1]
                
                    print(self.messageBuffer)
                
                    if len(commands) > 1:
                        
                        self.server.putCommands(commands)
        else:
            
            self.request.close()
                
    
    def setup(self):
        '''
        when creating a connection, this method is called. It registers this object at the server
        '''
        self.isConnected = self.server.connect(self)
        
        self.messageBuffer = ""
                
    
    def finish(self):
        '''
        when closing the connection, this method is called. It tells the server to disconnect
        '''
        self.server.disconnect(self)
        
    
    def send(self, message):
        '''
        sends message to the client
        '''
        try:
            self.request.sendall(message)
        
        except IOError, ioe:
            if ioe.errno == errno.EPIPE:
                
                self.server.log("broken pipe")
            else:
                raise ioe
