#!/usr/env/python
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

def init_chip():
    #remove recurrent connection and load learning biases
    nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_tbiocas15.biases')
    chip.configurator.set_parameter("PDPI_TAU_P", 2.4e-05)
    chip.configurator.set_parameter("PDPI_THR_P", 0.0)
    chip.configurator.set_parameter("VDPIE_THR_P", 0.0)
    chip.configurator.set_parameter("VDPIE_TAU_P", 2.0e-05)
    chip.configurator.set_parameter("VDPII_TAU_N", 2.0e-05)
    chip.configurator.set_parameter("VDPII_THR_N", 0.0)
    matrix_rec = np.zeros([256,256])
    time.sleep(0.2)
    nsetup.mapper._program_onchip_plastic_connections(matrix_rec)
    time.sleep(0.2)
    nsetup.mapper._program_onchip_programmable_connections(matrix_rec)
    #all syn 0
    time.sleep(0.2)
    nsetup.mapper._program_onchip_learning_state(matrix_rec)
    #average syn programmable leak with 50%exc and 50%inh, not really nedded but helps
    time.sleep(0.2)
    nsetup.mapper._program_onchip_exc_inh(np.random.choice([0,1],[256,256]))

def build_teaching_stimuli_br(fpre=50, duration=500, teach_freq = 120):
    '''
    this function returns the teaching spike train
    target all broadcast synapses
    teaching signal is on virtual syn
    '''
    a = range(256)
    syn = inputpop[a].synapses['broadcast'][0::256]
    stimulus_to_be_learnt = syn.spiketrains_poisson(fpre,duration=duration)
    #now add teaching signal (will excite the post neuron) 
    syn_virtual = inputpop.synapses['virtual_exc']
    stimulus_teach = syn_virtual.spiketrains_poisson(teach_freq, duration=duration)
    #final stim is teaching signal + actual stimulus
    stimulus = pyNCS.pyST.merge_sequencers(stimulus_teach,stimulus_to_be_learnt)

    return stimulus

def build_teaching_stimuli(neu=5, fpre=50, duration=500, teach_freq = 120): 
    '''
    this function returns the teaching spike train
    target only synapses of neuron neu
    teaching signal is on virtual syn
    '''
    index_neu = inputpop.synapses['learning'].addr['neu'] == neu    
    syn_this_neu = inputpop.synapses['learning'][index_neu]
    stimulus_to_be_learnt = syn_this_neu.spiketrains_poisson(fpre,duration=duration)
    #now add teaching signal (will excite the post neuron) 
    stimulus_teach = build_stimulus_virtual(neu, teach_freq, duration=duration)
    #final stim is teaching signal + actual stimulus
    stimulus = pyNCS.pyST.merge_sequencers(stimulus_teach,stimulus_to_be_learnt)

    return stimulus

def build_stimulus_virtual(neu, freq, duration=1000, t_start=0):
    index_neu = inputpop.synapses['virtual_exc'].addr['neu'] == neu
    syn_teaching_this_neu = inputpop.synapses['virtual_exc'][index_neu]
    stimulus_teach = syn_teaching_this_neu.spiketrains_poisson(int(freq), duration=duration, t_start=t_start)
    return stimulus_teach

def learn(stim, duration, n_presentations=5):
    #load learning parameters
    nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_tbiocas15.biases')
    chip.configurator.set_parameter("PDPI_TAU_P", 2.4e-05)
    chip.configurator.set_parameter("PDPI_THR_P", 0.0)
    chip.configurator.set_parameter("VDPIE_THR_P", 0.0)
    chip.configurator.set_parameter("VDPIE_TAU_P", 2.0e-05)
    chip.configurator.set_parameter("VDPII_TAU_N", 2.0e-05)
    chip.configurator.set_parameter("VDPII_THR_N", 0.0)
    #now present stimuli to be learn, together with teacher signal

    for i in range(n_presentations):
        out = nsetup.stimulate(stim,send_reset_event=False,duration=duration)

    #now freeze synapses to their state by increasing the drifts, and load testing biases.    
    nsetup.chips['mn256r1'].load_parameters('biases/biases_wijtesting_tbiocas15.biases')

    return out

def test():
    # test what you have learned...
    return
    
def read_syn_matrix_learning():
    #read the learnt state
    #NB it needs to delete recurrent connections ... plastic and not
    #matrix_rec = np.zeros([256,256])
    #nsetup.mapper._program_onchip_plastic_connections(matrix_rec)
    #nsetup.mapper._program_onchip_programmable_connections(matrix_rec)
    #time.sleep(0.2)
    sl.get_br()
    #time.sleep(0.2)

def fvirtual_vs_fpost(neu,fvirtual,duration=1000):
    '''
    measure fvirtual vs fpost
    '''
    ion()
    nsteps  = len(fvirtual)
    #build stimulus
    final_stim = []
    for this_step in range(nsteps):
        stim_p = build_stimulus_virtual(neu, fvirtual[this_step], duration=duration, t_start=duration*this_step)
        if(this_step == 0):
            final_stim = stim_p
        else:
            final_stim = pyNCS.pyST.merge_sequencers(final_stim,stim_p)

    out = nsetup.stimulate(final_stim,send_reset_event=False,duration=duration*nsteps+1000)
    fout = monitor.firing_rates(time_bin=duration)
    fin =  fvirtual
    figure()
    plot(fin[0:nsteps], fout[0:nsteps], 'o-', label='neu '+str(neu))
    xlabel('Freq [Hz] -> virtual syn')
    ylabel('Freq [Hz] -> post neu')
    legend(loc='best')
    
    return fin[0:nsteps], fout[0:nsteps]
 
def fvirtual_vs_fpost_all(fvirtual,duration=1000):
    '''
    measure fvirtual vs fpost
    '''
    ion()
    nsteps  = len(fvirtual)
    syn_virtual = inputpop.synapses['virtual_exc']
    #build stimulus
    final_stim = []
    for this_step in range(nsteps):
        stim_p  = syn_virtual.spiketrains_poisson(int(fvirtual[this_step]), duration=duration, t_start=duration*this_step)
        if(this_step == 0):
            final_stim = stim_p
        else:
            final_stim = pyNCS.pyST.merge_sequencers(final_stim,stim_p)

    out = nsetup.stimulate(final_stim,send_reset_event=False,duration=duration*nsteps+1000)
    fout = monitor.firing_rates(time_bin=duration)
    fin =  fvirtual
    figure()
    for i in range(256):
        plot(monitor.sl.firing_rate(time_bin=100)[i,:])
    xlabel('Freq [Hz] -> virtual syn')
    ylabel('Freq [Hz] -> post neu')
    figure()
    plot(fin[0:nsteps], fout[0:nsteps], 'o-', label='mean all neu')
    xlabel('Freq [Hz] -> virtual syn')
    ylabel('Freq [Hz] -> post neu')
    legend(loc='best')
    
    return fin[0:nsteps], fout[0:nsteps]    


#####################################################################
#simple experiment measuring ltp for all synapses of a single neuron
#####################################################################

init_chip()

#load biases
nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_tbiocas15.biases')

#enable clock neuron
print 'we enable clock neuron'
chip.configurator._set_neuron_tau1([255])
stim = build_stimulus_virtual(255,500,duration=1000)
nsetup.stimulate(stim, send_reset_event = False, duration=1000)

monitor = pyNCS.monitors.SpikeMonitor(inputpop.soma)
nsetup.monitors.import_monitors(monitor)

# find suitable post frequencies
fvirtual = np.linspace(10,55,5)
fin, fpost = fvirtual_vs_fpost_all(fvirtual,duration=1000)


#program broadcast learning syn
nsetup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
nsetup.mapper._program_onchip_broadcast_learning(np.ones([256,256]))

flower = sl.load_flower()

duration = 500
n_repetitions = 1
all_data_syn = []
post_rates = []
pre_rates = []
syn_pot = []
for this_trial_line in range(n_repetitions):

    #loop over post freq and measure number of pot syn at every presentation
    for this_trial in range(len(fvirtual)):
        
        #program broadcast learning syn
        nsetup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
        nsetup.mapper._program_onchip_broadcast_learning(flower)

        #set up the monitor
        monitor = pyNCS.monitors.SpikeMonitor(inputpop.soma)
        nsetup.monitors.import_monitors(monitor)
        #build stimulus 
        stim = build_teaching_stimuli_br(fpre=150,duration=duration,teach_freq=int(fvirtual[this_trial])) 
        print "################## run learning trial"
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_tbiocas15.biases')
        time.sleep(0.2)
        out = nsetup.stimulate(stim, duration=duration+200, send_reset_event=True)
        #out = learn(stim, duration+200, n_presentations=1)
        this_post_rates = monitor.sl.firing_rate(time_bin=duration)[:,0]
        print "################## reading syn matrix"
        sl.get_br()
        syn_learning = sl.state
        syn_state = syn_learning[0:256,:]
        syn_pot.append(np.sum(syn_state))
        #reset matrix learning and load biases
        nsetup.mapper._program_onchip_learning_state(np.zeros([256,256]))
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_tbiocas15.biases')
        #raw_input("plot")
        post_rates.append(this_post_rates)
        pre_rates.append(fvirtual[this_trial])
        all_data_syn.append(syn_state)


figure()
for i in range(len(fvirtual)):
    subplot(len(fvirtual),1,i+1)
    title("presentation number: "+str(i)+" tot pot syn: "+str(np.sum(all_data_syn[i])))
    imshow(np.rot90(np.rot90(all_data_syn[i])), interpolation="nearest")


