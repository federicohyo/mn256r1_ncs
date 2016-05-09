#!/usr/env/python
'''
author federico corradi
'''
import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('../api/wij/')
from wij import SynapsesLearning
import time
import scipy.signal
import subprocess
from pyNCS.pyST import SpikeList
ion()

#setup definitions

prefix='../'  
setuptype = '../setupfiles/mc_final_mn256r1_adcs.xml'
setupfile = '../setupfiles/final_mn256r1_adcs.xml'
nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
chip = nsetup.chips['mn256r1']

inputpop = pyNCS.Population('','')
inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  
broadcast_pop = pyNCS.Population('','')
broadcast_pop.populate_all(nsetup,'mn256r1', 'excitatory')  

nsetup.mapper._program_detail_mapping(2**0)

#init class synapses learning
sl = SynapsesLearning(inputpop, 'learning')

def build_stimulus_virtual(neu, freq, duration=1000, t_start=0):
    index_neu = inputpop.synapses['virtual_exc'].addr['neu'] == neu
    syn_teaching_this_neu = inputpop.synapses['virtual_exc'][index_neu]
    stimulus_teach = syn_teaching_this_neu.spiketrains_poisson(int(freq), duration=duration, t_start=t_start)
    return stimulus_teach
 
def upload_config(matrix_b, matrix_e_i, matrix_weight, matrix_learning_state, matrix_learning_recurrent, matrix_recurrent):
    '''
    upload config in the neuromorphic core
    ''' 
    nsetup.mapper._program_onchip_programmable_connections(matrix_recurrent)
    nsetup.mapper._program_onchip_broadcast_programmable(matrix_b)
    nsetup.mapper._program_onchip_weight_matrix_programmable(matrix_weight) #broadcast goes to weight 1 the rest is w 2
    nsetup.mapper._program_onchip_exc_inh(matrix_e_i)
    nsetup.mapper._program_onchip_learning_state(matrix_learning_state)
    nsetup.mapper._program_onchip_plastic_connections(matrix_learning_recurrent)     
    
    return
    
def remove_recurrent():
    nsetup.mapper._program_onchip_programmable_connections(np.zeros([256,256]))
    nsetup.mapper._program_onchip_plastic_connections(np.zeros([256,256]))
   
    return
       
def init_chip():
    #remove recurrent connection and load learning biases
    nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_spike_pattern.biases')

    matrix_b = np.random.choice([0,0,1],[256,256])
    matrix_e_i = np.random.choice([0,0,1,1,1],[256,256])
    matrix_e_i[0:6,:] = 0
    index_w_1 = np.where(matrix_b == 1)
    matrix_weight = np.zeros([256,256])
    matrix_weight[index_w_1] = 1
    index_w_2 = np.where(matrix_weight != 1)
    matrix_weight[index_w_2] = 2   
    matrix_recurrent = np.random.choice([0,0,1,1,1],[256,256])
    matrix_recurrent[0:6,:] = 0
    matrix_recurrent[:,0:6] = 0
    matrix_recurrent[index_w_1] = 0
    matrix_b[0:6,:] = 0
    
    matrix_learning_state = np.zeros([256,256])
    matrix_learning_recurrent = np.zeros([256,256])
    matrix_learning_recurrent[0:7,7:256] = 1

    upload_config(matrix_b, matrix_e_i, matrix_weight, matrix_learning_state, matrix_learning_recurrent, matrix_recurrent)
    
    chip.configurator._set_neuron_tau1([255])
    #enable clock neuron
    print 'we enable clock neuron'
    chip.configurator._set_neuron_tau1([255])
    stim = build_stimulus_virtual(255,500,duration=1000)
    nsetup.stimulate(stim, send_reset_event = False, duration=1000)
    
    return matrix_b, matrix_e_i, matrix_weight, matrix_learning_state, matrix_learning_recurrent, matrix_recurrent 

def build_teaching_stim():
    '''
    build stimuli with teaching signals
    ''' 
    
   
def show_conf(matrix_b, matrix_e_i, matrix_weight, matrix_learning_state, matrix_learning_recurrent, matrix_recurrent):
    '''
    plot configuration matrix
    '''       
    f, axarr = plt.subplots(4,2)
    axarr[0, 0].set_title("prog recurrent matrix")
    axarr[0, 0].imshow(matrix_recurrent,origin=(0,0))    
    axarr[1, 0].set_title("prog. w")
    axarr[1, 0].imshow(matrix_weight,origin=(0,0)) 
    axarr[2, 0].set_title("prog. exc_inh")
    axarr[2, 0].imshow(matrix_e_i,origin=(0,0)) 
    axarr[3, 0].set_title("prog. broadcast")
    axarr[3, 0].imshow(matrix_b,origin=(0,0)) 
    axarr[0, 1].set_title("learning state")
    axarr[0, 1].imshow(matrix_learning_state,origin=(0,0))
    axarr[1, 1].set_title("learning recurrent")
    axarr[1, 1].imshow(matrix_learning_recurrent,origin=(0,0))   
    
    return 
    

#####################################################################
#      simple experiment recognizing a spike pattern with MN256R1
#####################################################################

matrix_b, matrix_e_i, matrix_weight, matrix_learning_state, matrix_learning_recurrent, matrix_recurrent = init_chip()

duration_stim = 3500
f_stim = 250
f_teach = 80
teaching_at = 1500
teaching_duration = 100
n_teach_repetition = 3

#load biases
nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_spike_pattern.biases')
show_conf(matrix_b, matrix_e_i, matrix_weight, matrix_learning_state, matrix_learning_recurrent, matrix_recurrent)

#populations
pop_rcn = pyNCS.Population("","")
pop_rcn.populate_by_id(nsetup, 'mn256r1', 'excitatory', np.linspace(7,255,249))
pop_perceptrons = pyNCS.Population("","")
pop_perceptrons.populate_by_id(nsetup, 'mn256r1', 'excitatory', np.linspace(0,6,7))

#monitors
monitor_per = pyNCS.monitors.SpikeMonitor(pop_perceptrons.soma)
monitor_rcn = pyNCS.monitors.SpikeMonitor(pop_rcn.soma)
nsetup.monitors.import_monitors([monitor_per, monitor_rcn])

#buil virtual stimulus 
syn_virtual = pop_rcn.synapses['virtual_exc']
stim_virtual = syn_virtual.spiketrains_poisson(f_stim,duration=duration_stim)
#nsetup.stimulate(stim_virtual,send_reset_event=False)

#now build teaching signal
syn_teaching = pop_perceptrons.synapses['virtual_exc']
stim_teaching = syn_teaching.spiketrains_poisson(f_teach,t_start=teaching_at,duration=teaching_duration)

#put stimuli together
tot_teach = pyNCS.pyST.merge_sequencers(stim_virtual, stim_teaching)
tot_test = stim_virtual

for i in range(n_teach_repetition):
    out = nsetup.stimulate(tot_teach,send_reset_event=False)

nsetup.chips['mn256r1'].load_parameters('biases/biases_wijtesting_spike_pattern.biases')
out = nsetup.stimulate(tot_test,send_reset_event=False)
pyNCS.monitors.RasterPlot([monitor_per,monitor_rcn])

#read syn matrix
remove_recurrent()
sl.get_br()
#reload config
upload_config(matrix_b, matrix_e_i, matrix_weight, sl.state, matrix_learning_recurrent, matrix_recurrent)
nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_spike_pattern.biases')
