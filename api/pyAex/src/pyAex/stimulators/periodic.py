# -*- coding: utf-8 -*-
#General imports
import threading
import time
import numpy as np
import socket

#AER related imports
from pyAex import *
from pyNCS.pyST import *


class PeriodicStimulator(threading.Thread):
    '''
    This class can be used for doing continuous periodic stimulations
    '''
    def __init__(self, SeqChannelAddress=None, host='localhost', port_stim=50002, stim=None, period=1):
        '''
        *SeqChannelAddress:* Sequencer Channel Addressing object, if omitted, default channel is taken
        *host:* Monitoring and Sequencing AEX server hostname. (Must be same)
        *port_stim:* Port of the Monitoring Server (Default 50002)
        *stim:* Input stimulus
        *period:* Time period of repetation of stimulus in seconds
        '''
        threading.Thread.__init__(self)
        self.finished = threading.Event()
        if SeqChannelAddress == None:
            self.SeqChannelAddress = getDefaultSeqChannelAddress()
        else:
            self.SeqChannelAddress = SeqChannelAddress
        self.stim_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.stim_sock.connect((host, port_stim))
        self.period = period
        #Initiate the stimulus
        if stim != None:
            self.set_stimulus(stim)

    def __del__(self):
        self.stop()
        self.stim_sock.close()

    def stop(self):
        self.finished.set()

    def run(self):
        print('Starting stimulation')
        while self.finished.isSet() != True:
            start = time.time()
            self.stimonce()  # Stimulate
            elapsed = (time.time() - start)
            try:
                time.sleep(self.period - elapsed)
            except:
                print self.period, elapsed
        print('Stimulation stopped')
        self.finished.clear()

    def set_period(self, period):
        '''
        *period:* Period in seconds
        Please note. Period should be more than length of stimulus.
        '''
        self.period = period

    def set_stimulus(self, stim):
        '''
        Sets the periodic stimulus to be relayed.
        *stim:* stimulus should be a SpikeList compatible with pyAex's stimulate
        '''
        if stim.t_stop > self.period * 1000.:
            print("ERROR: Stimulus is larger than the 'period', set a larger period and retry.")
            return
        print "Multiplexing..."
        stimEvents = self.SeqChannelAddress.exportAER(stim, isi=True)
        evs_in = stimEvents.get_tmadev()
        self.stimByteStream = evs_in.tostring()
        print('Stimulus set')

    def stimonce(self):
        '''
        The function stimulates only once.
        '''
        self.stim_sock.send(self.stimByteStream)
