#!/usr/env/python
import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('/home/federico/projects/work/api/wij/')
sys.path.append('/home/federico/projects/work/api/perceptrons/')
import sys
sys.path.append('/home/federico/projects/work/api/classification_display/')
sys.path.append('/home/federico/projects/work/api/reservoir/')
sys.path.append('/home/federico/projects/work/api/retina_cx/')
from scipy import interpolate
import reservoir as L

from perceptrons import Perceptrons
from wij import SynapsesLearning
from retina_cx import Retina_cx
import time
import subprocess
from os import listdir
from os.path import isfile, join
import pickle

import missmatch as mm 

ion()

use_chip = True 


def learn(n_presentations=1)  :     
    '''
    Hamming distance plot
    '''
    #matrices = []
    for i in range(n_presentations):
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_ret_perceptrons_1.biases')
        chip.configurator.set_parameter("PDPI_TAU_P", 2.4e-05)
        chip.configurator.set_parameter("PDPI_THR_P", 0.0)
        chip.configurator.set_parameter("VDPIE_THR_P", 0.0)
        chip.configurator.set_parameter("VDPIE_TAU_P", 2.0e-05)
        chip.configurator.set_parameter("VDPII_TAU_N", 2.0e-05)
        chip.configurator.set_parameter("VDPII_THR_N", 0.0)
        
        win.train(xi,yi,1)   
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijtesting_ret_perceptrons.biases')
        #read_syn_matrix_learning()
        #net.upload_config(matrix_learning_state=False, matrix_exc_inh=True, matrix_programmable_w=False, matrix_programmable_rec=False, matrix_learning_rec=True)
        #matrices.append(sl.state)
    return #matrices

if use_chip:
    ##########################################
    # INIT PYNCS and CHIP
    ########################################

    prefix='/home/federico/projects/work/trunk/code/python/mn256r1_ncs/'  
    setuptype = '/home/federico/projects/work/trunk/code/python/mn256r1_ncs/setupfiles/mc_final_mn256r1_adcs.xml'
    setupfile = '/home/federico/projects/work/trunk/code/python/mn256r1_ncs/setupfiles/final_mn256r1_adcs.xml'
    nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
    chip = nsetup.chips['mn256r1']

    p = pyNCS.Population('', '')
    p.populate_all(nsetup, 'mn256r1', 'excitatory')

    inputpop = pyNCS.Population('','')
    inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  

    nsetup.mapper._program_detail_mapping(2**0) 

    #reset multiplexer
    features = Retina_cx(inputpop)
    features._init_fpga_mapper()
    
    nsetup.mapper._program_onchip_plastic_connections(np.zeros([256,256]))
    nsetup.mapper._program_onchip_learning_state(np.ones([256,256]))

    nsetup.mapper._program_onchip_recurrent(np.zeros([256,512]))
    nsetup.mapper._program_detail_mapping(0)  


    #set programmable synapse parameters
    chip.configurator.set_parameter("NPA_WEIGHT_INH0_N",0.001e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_INH1_N",1.2625e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_INH_N",0.03025e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_EXC_P",1.610925e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_EXC0_P",1.5398e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_EXC1_P",0.5e-6)
    chip.configurator.set_parameter("NPDPIE_THR_P",10e-12)
    chip.configurator.set_parameter("NPDPIE_TAU_P",80e-12)
    chip.configurator.set_parameter("NPDPII_THR_P",200e-12)
    chip.configurator.set_parameter("NPDPII_TAU_P",200e-12)

    #neuron parameters
    chip.configurator.set_parameter("IF_TAU2_N",8.6e-12)
    chip.configurator.set_parameter("IF_TAU1_N",0.033e-6)

    chip.configurator._set_all_neu_tau2()


    ##
    #use the broadcast
    a = range(256)
    broadcast_syn = features.pop_broadcast[a].synapses['broadcast'][0::256]
    matrix_b = np.ones([256,256])
    #matrix_b[0:6,:] = 0
    features.setup.mapper._program_onchip_broadcast_learning(matrix_b)

    #features.map_cx_output_layer_to_mn256r1(n_columns=256)

    #set neuron parameters
    #chip.configurator.set_parameter("IF_TAU2_N",3.3e-9)
    #chip.configurator.set_parameter("IF_DC_P",23.9e-11)
    #chip.configurator.set_parameter("VA_EXC_N",2.3e-5)
    #chip.configurator.set_parameter("VDPIE_TAU_P",82.0e-12)
    #chip.configurator.set_parameter("VDPIE_THR_P",82.0e-12)
    #chip.configurator.set_parameter("IF_THR_N",1000.0e-12)

    #chec if the neuron can get excited...
    #index_neu_zero_up = inputpop.synapses['virtual_exc'].addr['neu'] == 244
    #syn = inputpop.synapses['virtual_exc'][index_neu_zero_up]
    #spktrain = syn.spiketrains_regular(100)
    #nsetup.stimulate(spktrain,send_reset_event=False)

    #index_neu_zero_up = inputpop.synapses['programmable'].addr['neu'] == 0
    #syn = inputpop.synapses['programmable'][index_neu_zero_up]
    #spktrain = syn.spiketrains_poisson(600)
    #nsetup.stimulate(spktrain,send_reset_event=False)

    #neurons
    features_neu = np.linspace(0,0,1)
    perceptron_neu = np.linspace(1,255,127)

    #make on chip network
    feature_pop = pyNCS.Population('', '')
    perceptron_pop = pyNCS.Population('', '')
    feature_pop.populate_by_id(nsetup, 'mn256r1', 'excitatory', features_neu)
    perceptron_pop.populate_by_id(nsetup, 'mn256r1', 'excitatory', perceptron_neu)

    net = Perceptrons(perceptron_pop,feature_pop) 
    #net.matrix_learning_pot[:] = 0
    #net.upload_config()
        
    import display_datasets as dd
    my_data_folder = '/home/federico/Documents/ini/code/python/mn256r1_ncs/api/missmatch/data/'
    datasets = mm.seek_data(my_data_folder)
    x, y = mm.load_data(my_data_folder + "caltech101_airplanesVShelicopter.csv") 
    label_true = 6#2#4
    label_false = 50#72#20
    y_true = np.where(y == label_true, 1, 0)
    y_false = np.where(y == label_false, -1, 0)
    y_select = np.nonzero(y_true + y_false)[0]
    y = (y_true + y_false)[y_select]
    x = x[:, y_select]
    xi, yi, xt, yt = mm.sample_train_test(x, y, train_fraction=0.05,
                                          normalization=None)
    # pattern_shape is the shape of the original images
    win = dd.mainWindow(setup=nsetup, net = net,  pattern_shape=(128, 128),
                     show_shape=(360, 360),
                     roll_fac=3,
                     normalization=None, speed_fac=5)

    learn(1)


