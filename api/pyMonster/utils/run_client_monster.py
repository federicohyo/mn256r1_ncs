#!/usr/bin/env python

import sys
import time
import getopt
import socket
import threading

import numpy as np

from pyMonster import Client

def usage():
    print """
Usage: ./run_client_monster.py [options]

Options:
  -h, --help            show this help message and exit
  -a, --address         host IP address (default is "localhost")
  -p, --port            TCP port (default is 50001)
"""

def prompt_usage():
    print """ 
Possible commands are:
    q, CTRL+C           stop server and quit
    b <in_bits>         broadcast bits to the server
"""

def welcome():
        print """ 
        Welcome to the Monster configuration client!
        Use the prompt to send/receive messages.
        """

if __name__ == "__main__":

    opts, args = getopt.getopt(
        sys.argv[1:],
        "ha:p:",
        ["help", "address", "port"])

    host = "localhost"
    device = "/dev/aerfx2_0"
    port = 50003
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-a", "--address"):
            host = a
        elif o in ("-p", "--port"):
            port = int(a)

    print "Welcome!"
    print "\tHost", host
    print "\tPort", port
    my_client = Client(host=host, port=port)

    while 1:
        try:
            cmd = raw_input("> ")
            if cmd == 'q':
                print "Stopping client...",
                my_client.stop()
                print "done!"
                break
            elif len(cmd) > 1 and cmd[0] == 'b':
                msg = cmd.split()[1]
                my_client.send(msg)
            else:
                print "Unrecognized command."
                prompt_usage()
        except KeyboardInterrupt:
            print "Interrupted!"
            print "Stopping client...",
            my_client.stop()
            print "done!"
            break
