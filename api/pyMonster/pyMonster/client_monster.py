#!/usr/env/python

import sys
import time
import getopt
import socket
import threading
import struct

import numpy as np

BUFSIZE = 1024 * 8

class Client(socket.socket):

    def __init__(self, host, port):
        self.host = host
        self.port = port
        super(Client, self).__init__(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.connect((host, port))
        except socket.error as err:
            print err

    def send(self, data):
        self.sendall(data)

    def stop(self):
        self.close()

    def receive(self, BUFSIZE):
        print BUFSIZE
        received = self.recv(BUFSIZE)
        self.printout(received)
        return received

    def printout(self, data):
        print data
