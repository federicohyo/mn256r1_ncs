#!/usr/env/python

import os
import sys
import array
import stat
import time
import SocketServer
import threading
import getopt
import socket
import errno
import struct

import numpy as np

class Handler(SocketServer.BaseRequestHandler):

    def setup(self):
        self.server._add_client(self)

    def handle(self):
        while True:
            try:
                data = self.request.recv(1024)
                if data:
                    self.server._broadcast(data)
                    self.server._sequence(data)
                else:
                    break
            except socket.error, err:
                print err
                break

    def finish(self):
       # print "Disconnected:", self.client_address
        self.server._remove_client(self)


class Server(SocketServer.ThreadingMixIn, SocketServer.TCPServer, object):

    def __init__(self, device='/dev/aerfx2_0', host='localhost', port=50019):
        super(Server, self).__init__((host, port), Handler)
        self.device = device
        self.stop_me = False
        self.allow_reuse_address = True
        self._clients = dict()

    def start(self):
        while not self.stop_me:
            try:
                self.handle_request()
            except OSError as err:
                time.sleep(0.1)

    def stop(self):
        self.stop_me = True

    def get_clients(self):
        return self._clients.iterkeys()

    def _add_client(self, client):
        self._clients[client.client_address] = client

    def _remove_client(self, client):
        del self._clients[client.client_address]

    def _sequence(self, data):
        # data must look like final_address, e.g., 2151169671
        fd = os.open(self.device, os.O_WRONLY | os.O_NONBLOCK)
        #now here we need to uppack data in two 32 bits
        data = data.strip("[]")
        data = np.array(data.split(","), dtype="uint32")
        to_write = np.r_[int(data[0]), int(data[1])].astype('uint32').tostring()
        os.write(fd, to_write)
        os.close(fd)

    def _broadcast(self, data):
        try:
            for addr, cl in self._clients.iteritems():
                try:
                    cl.request.sendall(data)
                except socket.error, err:
                    if err[0] == errno.EPIPE:
                        print "Peer %s disconnected."%str(addr[0])
                    else:
                        print "Could not send to %s."%str(addr[0])
                        print err
        except RuntimeError:
            pass
