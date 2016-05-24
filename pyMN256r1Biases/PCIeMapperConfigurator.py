import math
import numpy as np
from pcieaex import Sequencer, Monitor


class PCIeMapperConfigurator():

    #PCIe
    pcieSeq = Sequencer
    pcieMon = Monitor
    pcieOpened = False

    def send_32(self, inbits):
        #send to fpga
        self.pcieSeq.put([inbits])

    def get_32(self):
        #read from fpga
        return self.pcieMon.get()

    def openPCIeSeq(self,SEQ_FPGA_ID,SEQ_FPGA_CHNL):
        print 'Opening Sequencer on PCIe (id=' + str(SEQ_FPGA_ID) + ', chnl=' + str(SEQ_FPGA_CHNL) + ') interface...'
        self.pcieSeq = Sequencer(SEQ_FPGA_ID, SEQ_FPGA_CHNL) #create sequencer

    def openPCIeMon(self,MON_FPGA_ID,MON_FPGA_CHNL):
        print 'Opening Monitor on PCIe (id=' + str(MON_FPGA_ID) + ', chnl=' + str(MON_FPGA_CHNL) + ') interface...'
        self.pcieMon = Monitor(MON_FPGA_ID, MON_FPGA_CHNL,run=True) #create Monitor

    def closePCIeSeq(self):
        print 'Closing Sequencer PCIe interface...'
        del self.pcieSeq

    def closePCIeMon(self):
        print 'Closing Monitor PCIe interface...'
        self.pcieMon.stop()
        del self.pcieMon