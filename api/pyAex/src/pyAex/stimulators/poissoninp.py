# -*- coding: utf-8 -*-
#AER related imports
from pyAex.stimulators import PeriodicStimulator


class PoissonStimulator(PeriodicStimulator):
        '''
        This class can be used for doing continuous periodic stimulations
        '''
        def __init__(self, grp, rate = None, SeqChannelAddress=None, host='localhost', port_stim=50002): 
                '''
                *grp:* Address group of neurons generating the spikes
                *rate:* Mean rate per neuron
                *SeqChannelAddress:* Sequencer Channel Addressing object, if omitted, default channel is taken
                *host:* Monitoring and Sequencing AEX server hostname. (Must be same)
                *port_stim:* Port of the Monitoring Server (Default 50002)
                '''

                PeriodicStimulator.__init__(self,
                                            SeqChannelAddress=SeqChannelAddress,
                                            host=host, port_stim=port_stim)
                self.grp = grp
                self.rate = rate

        def set_stimulus(self, rate):
                '''
                Sets the periodic stimulus to be relayed.
                *rate:* The firing rate that needs to be set
                '''
                self.rate = rate

        @property
        def stimByteStream(self):
                stim = self.grp.spiketrains_poisson(self.rate,
                                                    duration=self.period*1000.)
                stimEvents = self.SeqChannelAddress.exportAER(stim,isi=True)
                return stimEvents.get_tmadev().tostring()
        
