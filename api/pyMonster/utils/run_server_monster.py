#!/usr/bin/env python

import os
import sys
import stat
import time
import SocketServer
import threading
import getopt

import numpy as np

from pyMonster import Server


def usage():
    print """
Usage: ./run_server_monster.py [options]

Options:
  -h, --help            show this help message and exit
  -d, --device          device file (default is '/dev/aerfx2_0')
  -p, --port            TCP port (default is 50001)
  -i, --ip              host IP (default is "localhost")
"""


def welcome():
    print """
Welcome to the Monster configuration server!
The server is running.
"""


def prompt_usage():
    print """
Possible commands are:
    q, CTRL+C           stop server and quit
    w                   list connected clients
    s <string>          send string to device
    b <message>         broadcast a message to clients
"""

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "hi:p:d:", ["help", "ip", "port", "device"])
    except getopt.GetoptError as err:
        print err
        usage()
        sys.exit()

    host = "localhost"
    device = "/dev/aerfx2_0"
    port = 50019
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-i", "--ip"):
            host = a
        elif o in ("-p", "--port"):
            port = int(a)
        elif o in ("-d", "--device"):
            device = a

    # start server
    server = Server(device=device, host=host, port=port)
    start_th = threading.Thread(target=server.start)
    start_th.setDaemon(True)
    start_th.start()
    serve_th = threading.Thread(target=server.serve_forever)
    serve_th.setDaemon(True)
    serve_th.start()

    welcome()
    prompt_usage()

    # make it interactive
    while 1:
        try:
            cmd = raw_input("> ")
            if cmd == 'q':
                print "Stopping server...",
                server.stop()
                print "done!"
                break
            elif len(cmd) > 1 and cmd[0] == 'b':
                msg = cmd.split()[1]
                server._broadcast(msg)
            #elif len(cmd) > 1 and cmd[0] == 's':
                #msg = cmd.split()[1]
                #server._sequence(msg)
            elif cmd == 'w':
                print "Connected clients:"
                clients = server.get_clients()
                for c in clients:
                    print "\t", c[0]
            else:
                print "Unrecognized command."
                prompt_usage()
        except KeyboardInterrupt:
            print "Interrupted!"
            print "Stopping server...",
            server.stop()
            print "done!"
            break

    # shutdown stuff
    print "Shutting down...",
    server.shutdown()
    print "done!"
    print " -- Ciao"
