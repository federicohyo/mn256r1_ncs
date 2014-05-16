# Import libraries
from pylab import *
import numpy as np
import biasusb_wrap

dpiNames = {'NPDPIE': 0,
            'NPDPII': 1,
            'PDPIE': 2,
            'VDPIE': 3,
            'VDPII': 4
}

plasticLatches = {'Recurrent': 0,
                  'Broadcast': 1
}

non_plasticLatches = {'Recurrent': 0,
                  'Exc_Inh': 1,
                  'W0': 2,
                  'W1': 3,
                  'Broadcast': 4
}


def set_plastic_weight(neu,syn,value):
    '''
        this function set plastic weight to value (1/0) of row (neu) and column (synapse) 
    '''
    if(value == 1):
        vv = 1*256*256+syn*256+neu+1051136
        biasusb_wrap.send_32(int(vv))
        #print vv
    elif(value == 0):
        vv = syn*256+neu+1051136
        #print vv
        biasusb_wrap.send_32(int(vv))

def set_recurrent_bit(neu,syn,onoff,recurrent = 0):
    '''
        this function set recurrent bit of row (neu) and col (syn) to on onoff (1 or 0)
    '''
    plastic = False
    programmable = False
    #matrix = matrix.astype(int)
    if (syn < 256):
        plastic = True
    else:
        programmable = True
    if(plastic):
        offset = 788992
        matrix_value = onoff
        value = matrix_value*256*256+syn*256+neu+offset
    if(programmable):
        offset = 133632
        matrix_value = onoff
        value = matrix_value*256*256+(syn-256)*256+neu+offset
    #print value
    biasusb_wrap.send_32(int(value))



def set_recurrent_bit_plastic_syn(neu,syn,onoff,recurrent = 0):
    '''
        this function set recurrent bit of row (neu) and col (syn) to on onoff (1 or 0)
    '''
    biasusb_wrap.send_32(recurrent*2*256*256+onoff*256*256+syn*256+neu+788992)


def set_connections_matrix_programmable(matrix):
    '''
        this function set recurrent matrix of programmable synapses shape(256,256) 256-> neu ,  256->syn
    '''
    for col in range(256,512):
        for row in range(256):
	    load_recurrent_bit(row,col,matrix[row,col-256],recurrent = 0)


def set_connections_matrix_all(matrix):
    '''
        this function set recurrent matrix of both plastic and programmable synapses shape(256,512) 256-> neu ,  512->syn
    '''
    plastic = False
    programmable = False
    #addr = []
    if( np.shape(matrix) != (256,512) ):
        print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,512)'
        return 0
    else:
        #matrix = matrix.astype(int)
        for col in range(512):
            for row in range(256):
                load_recurrent_bit(row,col,matrix[row,col],recurrent = 0)


def set_connections_matrix_plastic(matrix):
    '''
        this function set recurrent matrix 
    '''
    if( np.shape(matrix) != (256,256) ):
        print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
        return 0
    else:
        #matrix = matrix.astype(int)
        for col in range(256):
            for row in range(256):
                load_recurrent_bit(row,col,matrix[row,col],recurrent = 0)

def set_weight_matrix_plastic(matrix):
    '''
        this function set plastic matrix weights (only binary)
    '''
    if( np.shape(matrix) != (256,256) ):
        print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
        return 0
    else:
        #matrix = matrix.astype(int)
        for row in range(256):
            for col in range(256):
                #value = matrix[row,col]*256*256+col*256+row+1051136
                load_plastic_weight(row,col,matrix[row,col])
                #biasusb_wrap.send_32(int(value))

def set_programmable_exc_inh(neu,syn,ei):
    '''
        this function program the programmable synapse to be exc (ei=1) or inh (ei=0)
    '''
    value = 1*2*256*256+ei*256*256+syn*256+neu+133632
    biasusb_wrap.send_32(int(value)) 

def set_matrix_exc_inh(matrix):
    '''
        this function set the programmable synapses to be exc or inhibitory input np.shape(256,256) | 0 = inh / 1 = exc
    '''
    if( np.shape(matrix) != (256,256) ):
        print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
        return 0
    else:
        for neu in range(256):
            for syn in range(256):
                load_programmable_exc_inh(neu,syn,matrix[neu,syn])

def set_weight_programmable(neu,syn,w):
    '''
        this function set programmable weights (4 values one of [0,1,2,3] )
    '''
    if(w > 3):
        print "error weights!! please enter w in range [0,3]"
        raise Exception
    if w == 0:
        bit0 = 0
        bit1 = 0
    if w == 1:
        bit0 = 1
        bit1 = 0
    if w == 2:
        bit0 = 0
        bit1 = 1
    if w == 3:
        bit0 = 1
        bit1 = 1


    value = 2*2*256*256+bit0*256*256+syn*256+neu+133632
    biasusb_wrap.send_32(int(value))
    #print value
    value = 3*2*256*256+bit1*256*256+syn*256+neu+133632
    biasusb_wrap.send_32(int(value))
    #print value

def set_weight_matrix_programmable(matrix):
    '''
        this function set programmable matrix weights (4 values one of [0,1,2,3] )
    '''
    if( np.shape(matrix) != (256,256) ):
        print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
        return 0
    else:
        for neu in range(256):
            for syn in range(256):
                load_weight_programmable(neu,syn,matrix[neu,syn])


def set_broadcast_matrix(matrix):
    '''
        this function set broadcast bit (0 / 1) (shape 256,512)
    '''
    if( np.shape(matrix) != (256,512) ):
        print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,512)'
        return 0
    else:
        for row in range(256):
            for col in range(256):
                if(col < 256):
                    value = 1*2*256*256+matrix[row,col]*256*256+col*256+row+788992
                else:
                    value = 4*2*256*256+matrix[row,col]*256*256+col*256+row+133632
                biasusb_wrap.send_32(int(value))


def set_neuron_tau_1(neurons):
    '''
        set neurons to tau 1 ; neurons is an array 
    '''
    for i in range(len(neurons)):
        biasusb_wrap.send_32(int(neurons[i]+1249026))
        biasusb_wrap.send_32(int(1249283))                     
        biasusb_wrap.send_32(int(neurons[i]+1249026))


def set_neurons_tau_2():
    '''
        set all neuron to tau 2
    '''
    biasusb_wrap.send_32(1249282)
    biasusb_wrap.send_32(1249283)


def setMonitorDPI(name,neuron):
	biasusb_wrap.send_32(int(dpiNames[name])*256+neuron+1247744)


def setMonitorNeuron(neuron):
    biasusb_wrap.send_32(1249024)
    biasusb_wrap.send_32(1249025)
    biasusb_wrap.send_32(neuron+1249284) #PLEASE FIX!
    biasusb_wrap.send_32(1249025)
    biasusb_wrap.send_32(neuron+1249284)
