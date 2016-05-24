import math
import numpy as np
import socket
#from pcieaex import Sequencer, Monitor

import sys
#sys.path.append('/home/federico/projects/work/trunk/code/python/spkInt/scripts')
#sys.path.append('/home/federico/project/work/trunk/code/python/spkInt/scripts/mapperlib')
sys.path.append('biasgenlib')
import biasusb_wrap

class MN256R1Configurator():

    #default file
    defaultfile = 'biases/defaultBiases.txt'

    #PCIe
    #pcieSeq = Sequencer
    #pcieMon = Monitor
    pcieOpened = False
    
    #TCP
    tcpOpened = False

    biasNames = {"IF_RST_N": 0,
                 "IF_BUF_P": 1,
                 "IF_ATHR_N": 2,
                 "IF_RFR1_N": 3,
                 "IF_RFR2_N": 4,
                 "IF_AHW_P": 5,
                 "IF_AHTAU_N": 6,
                 "IF_DC_P": 7,
                 "IF_TAU2_N": 8,
                 "IF_TAU1_N": 9,
                 "IF_NMDA_N": 10,
                 "IF_CASC_N": 11,
                 "IF_THR_N": 12,
                 "SL_THDN_P": 13,
                 "SL_MEMTHR_N": 14,
                 "SL_BUF_N": 15,
                 "SL_THMIN_N": 16,
                 "SL_WTA_P": 17,
                 "SL_CATHR_P": 18,
                 "SL_THUP_P": 19,
                 "SL_CATAU_P": 20,
                 "SL_CAW_N": 21,
                 "VA_INH_P": 22,
                 "VDPII_TAU_N": 23,
                 "VDPII_THR_N": 24,
                 "VA_EXC_N": 25,
                 "VDPIE_TAU_P": 26,
                 "VDPIE_THR_P": 27,
                 "FB_REF_P": 28,
                 "FB_WTA_N": 29,
                 "FB_BUF_P": 30,
                 "FB_CASC_N": 31,
                 "FB_INVERSE_TAIL_N": 32,
                 "FB_INVERSE_REF_N": 33,
                 "PDPI_BUF_N": 34,
                 "PDPI_VMONPU_P": 35,
                 "PDPI_TAU_P": 36,
                 "PDPI_THR_P": 37,
                 "NPDPIE_THR_P": 38,
                 "NPDPIE_TAU_P": 39,
                 "NPDPII_TAU_P": 40,
                 "NPDPII_THR_P": 41,
                 "NPA_WEIGHT_STD_N": 42,
                 "NPA_WEIGHT_INH0_N": 43,
                 "NPA_PWLK_P": 44,
                 "NPA_WEIGHT_INH1_N": 45,
                 "NPA_WEIGHT_EXC_P": 46,
                 "NPA_WEIGHT_EXC1_P": 47,
                 "NPA_WEIGHT_EXC0_P": 48,
                 "NPA_WEIGHT_INH_N": 49,
                 "PA_WDRIFTDN_N": 50,
                 "PA_WDRIFTUP_P": 51,
                 "PA_DELTAUP_P": 52,
                 "PA_DELTADN_N": 53,
                 "PA_WHIDN_N": 54,
                 "PA_WTHR_P": 55,
                 "PA_WDRIFT_P": 56,
                 "PA_PWLK_P": 57,
                 "BIAS_58": 58,
                 "BIAS_59": 59,
                 "BIAS_60": 60,
                 "BIAS_61": 61,
                 "BIAS_62": 62,
                 "BIAS_63": 63,
                 "BUFFER_BIASES": 64,
                 "SSP": 65,
                 "SSN": 66}

    biasBits = {"LowBias": 0,
                "HighBias": 1,
                "CascodeBias": 0,
                "Normal": 1,
                "PBias": 0,
                "NBias": 1,
                "BiasDisable": 0,
                "BiasEnable": 1,
                "NONE": 0}

    biasCoarseCurrents = {"24u": 0,
                          "3.2u": 1,
                          "0.4u": 2,
                          "50n": 3,
                          "6.5n": 4,
                          "820p": 5,
                          "105p": 6,
                          "15p": 7,
                          "SPECIAL": 7}


    def convertBias(self,biasName, coarse, fine, biasLowHigh, biasCascode, biasType, biasEnable):
        #adddes
        addr = self.biasNames[biasName]
        #configuration bits
        confBits = self.biasBits[biasLowHigh] << 3 | self.biasBits[biasCascode] << 2 | self.biasBits[biasType] << 1 | self.biasBits[biasEnable]
        #ssn and ssp
        if addr == 65 or addr == 66:
            inbits = addr << 16 | 63 << 10 | fine << 4 | confBits
        else:
            special = 0
            #bit pattern
            coarse_reversed = sum(1 << (2 - i) for i in range(3) if coarse >> i & 1)
            inbits = addr << 16 | special << 15 | coarse_reversed << 12 | fine << 4 | confBits
        return inbits


    def findCoarseAndFine(self,biasValue):
        if biasValue < math.pow(8, -5):
            biasValue = math.pow(8, -5)
        elif biasValue > math.pow(8, 2):
            biasValue = math.pow(8, 2)
        multiplier = math.ceil(math.log(biasValue, 8))
        coarse = int(2 - multiplier)
        maxCurrent = math.pow(8, 2 - coarse)
        fine = int(round(biasValue / maxCurrent * 256))
        if fine >= 255:
            fine = 255
        corrected = fine / 256.0 * maxCurrent
        biasCoarseFineCorrected = [coarse, fine, corrected]
        return biasCoarseFineCorrected


    ## load default txt bias file
    def load_default(self,def_file=defaultfile):
        biases = []
        #read from bias file
        with open(def_file) as f:
            for line in f:
                biases.append(line.split(','))

        return biases


    def save_all_biases(self,txtfile, output):
        print 'Saving biases to \''+txtfile+'\'\n'
        # Open file
        fo = open(txtfile, "wb")
        fo.write(output);
        # Close opened file
        fo.close()
        return 0


    def read_default_bias(self,biasName, def_file=defaultfile):
        biases = self.load_default(def_file)

        for i in range(len(biases)):
            if (biasName == biases[i][0]):
                bias = biases[i]

        return bias


    def set_bias(self,biasName, valueCoarse, valueFine, biasLowHigh='', biasType='', biasCascode='', biasEnable=''):
        biases = self.load_default(self.defaultfile)
        for i in range(len(biases)):
            if (biasName == biases[i][0]):
                if (biasLowHigh == ''):
                    biasLowHigh = biases[i][3]
                if (biasType == ''):
                    biasType = biases[i][5]
                if (biasCascode == ''):
                    biasCascode = biases[i][4]
                if (biasEnable == ''):
                    biasEnable = biases[i][6]


        #get coarse and fine values
        #biasCoarseFineCorrected = findCoarseAndFine(float(value))
        #coarse = biasCoarseFineCorrected[0]
        #fine = biasCoarseFineCorrected[1]
        #biasValueCorrected = biasCoarseFineCorrected[2]

        inbits = self.convertBias(biasName, self.biasCoarseCurrents[valueCoarse], valueFine, biasLowHigh, biasCascode, biasType,biasEnable)
        printbits = '{:023b}'.format(inbits)
        print biasName + ", " + valueCoarse + " (" + str(valueFine) + ")" + ", " + biasLowHigh + ", " + biasCascode + ", " + biasType + ", " + biasEnable + " -> " + printbits[0:7] + ' ' + printbits[7:24]

        #bias command has first two bits "11"
        self.send_32(inbits)# + (3<<30))
        
        #send bias by TCP
        if self.tcpOpened:
            self.tcps.send(biasName+" "+valueCoarse+" "+str(valueFine))

    def send_32(self, inbits):
        #send to fpga
        #self.pcieSeq.put([inbits])
        #if self.pcieMon.get() == inbits:
        #    print "[OK]"
        #else:
        #print "[ERROR]"
        print "inbits", inbits
        programming_bits = np.bitwise_and(inbits,65535) #bits 0 16 
        address_branch = (np.bitwise_and(inbits,8323072)>>16) #bits 17 to 22 
        print "send stuff"
        print "address_branch", np.binary_repr(address_branch)
        print "programming_bits", np.binary_repr(programming_bits)
        print "address_branch", (address_branch)
        print "programming_bits", (programming_bits<<7)
        final_address = (programming_bits<<7) + (address_branch) + 2**31
        print "final address", final_address
        biasusb_wrap.send_32(int(final_address))


    def set_aer(self, address):
        print "Sending AER address: " + '{:021b}'.format(address) + ' (' + str(address) + ')'
        #aer command has to first two bits "10"
        #self.send_32(address + (2<<30))
        biasusb_wrap.send_32(address)

    def set_conf(self, address):
        print "Sending Configuration address: " + '{:021b}'.format(address) + ' (' + str(address) + ')'
        #conf command has to first two bits "01"
        #self.send_32(address + (1<<30))
        #biasusb_wrap.send_32_1(address+(1<<30))
        biasusb_wrap.send_32(address)

    def openPCIe(self,FPGA_ID,FPGA_CHNL):
        print 'Opening PCIe (id=' + str(FPGA_ID) + ', chnl=' + str(FPGA_CHNL) + ') interface...'
        #self.pcieSeq = Sequencer(FPGA_ID, FPGA_CHNL) #create sequencer
        #self.pcieMon = Monitor(FPGA_ID, FPGA_CHNL,run=True) #create Monitor

    def closePCIe(self):
        print 'Closing PCIe interface...'
        #self.pcieMon.stop()
        #del self.pcieSeq
        #del self.pcieMon
        
    def openTCP(self,TCP_IP,TCP_PORT):
    	self.tcps = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print 'Connecting TCP socket...'
        self.tcpOpened = True
        self.tcps.connect((TCP_IP, TCP_PORT))
        
    def closeTCP(self):
        print 'Closing TCP socket...'
        self.tcpOpened = False
        self.tcps.close()
