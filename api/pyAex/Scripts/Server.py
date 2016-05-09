#!/usr/bin/python

'''
AexServer -- Example script

 @author: Daniel Sonnleithner
 @contact: daniel.sonnleithner@ini.phys.ethz.ch
 @organization: Institute of Neuroinformatics

 @copyright: (c) by Daniel Sonnleithner
 
 Created on: Aug 19, 2009
 
 @version: 0.1
 
 Change log:
'''

from pyAexServer import start_server
import getopt, sys

if __name__=='__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:m:s:c:d", ["filename=", "monport=", "seqport=", "ctrlport=", "debug"])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)
    dev_file = '/dev/aerfx2_0'
    debug = False
    mon_port=50001
    seq_port=50002
    ctrl_port=50003
    for o, value in opts:
        if o == "-d":
            debug = True
        elif o in ("-f", "--filename"):
            dev_file=value
	elif o in ("-m", "--monport"):
	    mon_port=int(value)
	elif o in ("-s", "--seqport"):
	    seq_port=int(value)
	elif o in ("-c", "--ctrlport"):
	    ctrl_port=int(value)
        else:
            assert False, "unhandled option, %s %s"%(o, value)



    start_server(dev_file, monPort=mon_port, stimPort=seq_port, ctrlPort= ctrl_port, prompt=True, debug = debug)
