# -*- coding: utf-8 -*-
#General imports
import threading
import time
import socket
import NeuroTools.stgen as stgen
from NeuroTools.signals import SpikeList

#AER related imports
from pyAex import *
from pyNCS import pyST

import numpy as np

class PoissonStimulator(threading.Thread):
        '''
        This class can be used for doing continuous periodic stimulations
        '''
        def __init__(self, SeqChannelAddress=None, channel=0, host='localhost', port_stim=50002,
                     rates=None, period=1, seq_export=True,
                     load_stim_filename=None): 
                '''
                *SeqChannelAddress:* Sequencer Channel Addressing object, if omitted, default channel is taken
                *host:* Monitoring and Sequencing AEX server hostname. (Must be same)
                *port_stim:* Port of the Monitoring Server (Default 50002)
                *rates:* Pairs of logical/physical addresses corresponding rate [addr, rate]
                *seq_export:* If True, use logical addresses and exportAER stuff
                *load_stim_filename:* If not None, stimByteStream is loaded from that file
                '''

                #threading.Thread.__init__ ( self )
                super(PoissonStimulator, self).__init__()
                #self.finished=threading.Event()
                self.is_running = True
                if SeqChannelAddress==None:
                        self.SeqChannelAddress=pyST.getDefaultSeqChannelAddress()
                else:
                        self.SeqChannelAddress=SeqChannelAddress
                

                #self.aexfd = os.open("/dev/aerfx2_0", os.O_RDWR | os.O_NONBLOCK)
                self.stim_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.stim_sock.connect((host, port_stim))
                
                self.period = period #period in seconds
                        
                self.sg = stgen.StGen()
                
                self.channel = channel
                self.rates = rates

                self.seq_export = seq_export

                self.load_stim_filename = load_stim_filename

        def __del__(self):                
                self.stop()
                self.stim_sock.close()

        def stop(self):
                #self.finished.set()
                self.is_running = False 

        def run(self):
                #print('Starting stimulation')
                #while self.finished.isSet() != True:
                while self.is_running:
                        start = time.time()
                        self.stimonce() #Stimulate
                        elapsed = (time.time() - start)
                        try:
                                time.sleep(self.period-elapsed)
                        except:
                                continue
                                #print self.period, elapsed
                #print('Stimulation stopped')
        
       
        def set_stimulus(self, addr):
                '''
                Sets the periodic stimulus to be relayed.
                *addr:* This is a dict of the form {channel:[[addr,rate]]}
                '''
                self.addr = addr
                
        
        def stimonce(self):
                '''
                The function stimulates only once.
                '''
                if self.load_stim_filename is not None:
                    self.stimByteStream = np.load(self.load_stim_filename)
                else:
                    stim = {}
                    stimlist = SpikeList([],[])
                    for adr, rate in self.rates:
                        stimlist[adr] = self.sg.poisson_generator(rate=rate,
                                                                  t_stop=self.period*1000.)
                        stim[self.channel] = stimlist        
                    if self.seq_export:
                        # translate logical -> physical
                        stimEvents = self.SeqChannelAddress.exportAER(stim,isi=True)
                    else:
                        multi = []
                        # concatenate events
                        for id, train in stimlist.spiketrains.iteritems():
                            for i, t in zip(np.repeat(id, len(train.spike_times)),
                                            train.spike_times):
                                multi.append([i, t*1000])
                        # multiplex
                        try:
                            multi = np.r_[multi]
                            multi = multi[np.argsort(multi[:, 1])]
                            multi[:, 1] = np.concatenate([[0], np.diff(multi[:, 1])])
                        except IndexError:
                            multi = []
                        # create events instance
                        stimEvents = pyST.events(multi)
                    self.stimByteStream=stimEvents.get_tmadev().tostring()
                self.stim_sock.send(self.stimByteStream)
