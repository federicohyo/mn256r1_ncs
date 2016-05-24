import math
import numpy as np
from pcieaex import Sequencer, Monitor


class DVSPAIRConfigurator():

    #PCIe
    pcieSeq = Sequencer
    pcieOpened = False

    def send_32(self, inbits):
        #send to fpga
        self.pcieSeq.put([inbits])

    def send_conf_params(self, bits):
        print "Sending Configuration bits: " + '{:05b}'.format(bits)
        self.send_32(bits + (2<<28))

    def send_left_params(self, bits):
        print "Sending left filter parameter bits: " + '{:028b}'.format(bits)
        self.send_32(bits + (1<<28))

    def send_right_params(self, bits):
        print "Sending right filter parameter bits: " + '{:028b}'.format(bits)
        self.send_32(bits + (0<<28))

    def openPCIe(self,FPGA_ID,FPGA_CHNL):
        print 'Opening PCIe (id=' + str(FPGA_ID) + ', chnl=' + str(FPGA_CHNL) + ') interface...'
        self.pcieSeq = Sequencer(FPGA_ID, FPGA_CHNL) #create sequencer

    def closePCIe(self):
        print 'Closing PCIe interface...'
        del self.pcieSeq