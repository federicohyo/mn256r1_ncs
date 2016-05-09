#!/usr/env/python
import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('../api/wij/')
sys.path.append('../api/bioamp/')
sys.path.append('../api/perceptrons/')
import sys
sys.path.append('/home/federico/projects/work/trunk/code/python/spkInt/scripts/')
import functions
sys.path.append('../api/reservoir/')
sys.path.append('../api/retina/')
sys.path.append('../gui/reservoir_display/')
from scipy import interpolate
import reservoir as L

from perceptrons import Perceptrons
from wij import SynapsesLearning
from bioamp import Bioamp
import time
import scipy.signal
import subprocess
from pyNCS.pyST import SpikeList


res = L.Reservoir() #reservoir without configuring the chip 
ion()
figs = figure()

prefix='../'  
setuptype = '../setupfiles/mc_final_mn256r1_adcs.xml'
setupfile = '../setupfiles/final_mn256r1_adcs.xml'
nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
chip = nsetup.chips['mn256r1']

p = pyNCS.Population('', '')
p.populate_all(nsetup, 'mn256r1', 'excitatory')

inputpop = pyNCS.Population('','')
inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  
#then load bioamp biases on top of these
rcnpop = pyNCS.Population('neurons', 'for fun') 
rcnpop.populate_all(nsetup,'mn256r1','excitatory')
broadcast_pop = pyNCS.Population('','')
broadcast_pop.populate_all(nsetup,'mn256r1', 'excitatory')  

perceptron_pops = pyNCS.Population("","")
perceptron_pops .populate_by_id(nsetup,'mn256r1', 'excitatory',np.linspace(0,5,6))  
out_perc = pyNCS.monitors.SpikeMonitor(perceptron_pops.soma)

rcn_pops = pyNCS.Population("","")
rcn_pops .populate_by_id(nsetup,'mn256r1', 'excitatory',np.linspace(7,254,248))  
out_rcn = pyNCS.monitors.SpikeMonitor(rcn_pops.soma)

nsetup.monitors.import_monitors([out_rcn, out_perc])

#reset multiplexer
bioamp = Bioamp(inputpop)
bioamp._init_fpga_mapper()

#####################
# RECONSTRUCT SIGNAL
#####################
def go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.0725, do_plot=True, do_detrend=False):
            
    #out = nsetup.stimulate({},duration=duration_rec)
    raw_data = out[0].raw_data()
    index_dn = np.where(raw_data[:,1] == dnch)[0]
    index_up = np.where(raw_data[:,1] == upch)[0]

    raw_data_input = []
    raw_data_input.extend(raw_data[index_dn,:])
    raw_data_input.extend(raw_data[index_up,:])
    raw_data_input = np.reshape(raw_data_input,[len(index_dn)+len(index_up),2])

    raw_data = np.delete(raw_data, index_dn,axis=0)
    raw_data = np.delete(raw_data, index_up,axis=0)

    index_sort = np.argsort(raw_data_input[:,0])   
    #reconstruct from up and down
    up_index = np.where(raw_data_input[:,1]==upch)
    dn_index = np.where(raw_data_input[:,1]==dnch)
    index_ord =  np.argsort(raw_data_input[:,0])   
    signal = np.zeros([len(raw_data_input)])
    for i in range(1,len(raw_data_input)):
        if(raw_data_input[index_ord[i],1] == upch):
            signal[i] = signal[i-1] + delta_up
        if(raw_data_input[index_ord[i],1] == dnch):
            signal[i] = signal[i-1] - delta_dn
            
    signal_trace_good = [raw_data_input[index_ord,0],signal]
    if do_detrend:
        df = scipy.signal.detrend(signal_trace_good[1])
        signal_trace_good  = np.array([signal_trace_good[0],df]).transpose()
    else:
        signal_trace_good  = np.array([signal_trace_good[0],signal_trace_good[1]]).transpose()  
        
    if do_plot == True:
        figure(figs.number)
        plot(signal_trace_good[:,0],signal_trace_good[:,1])
        
    return signal_trace_good
        
##### very cool
# now we build a binary classifier
###################################
################################################
################################################
# we need to load the stimulus, record and add a teachear signal
# we need to build the network



####configure chip matrixes
print "configure chip matrixes"
matrix_b = np.random.choice([0,0,1],[256,256])
#matrix_b[0:6,:] = 0
matrix_b[:,0:6] = 0

matrix_e_i = np.random.choice([0,0,1,1,1],[256,256])
index_w_1 = np.where(matrix_b == 1)
matrix_weight = np.zeros([256,256])
matrix_weight[index_w_1] = 1
index_w_2 = np.where(matrix_weight != 1)
matrix_weight[index_w_2] = 2
matrix_recurrent = np.random.choice([0,0,1,1,1],[256,256])
matrix_recurrent[index_w_1] = 0
matrix_recurrent[:,0:6] = 0
nsetup.mapper._program_onchip_programmable_connections(matrix_recurrent)
nsetup.mapper._program_onchip_broadcast_programmable(matrix_b)
nsetup.mapper._program_onchip_weight_matrix_programmable(matrix_weight) #broadcast goes to weight 1 the rest is w 2
nsetup.mapper._program_onchip_exc_inh(matrix_e_i)
# connect all neurons to the first 6 neurons via learning synapses
matrix_learning = np.zeros([256,256])
matrix_learning[:,0:6] = 1
nsetup.mapper._program_onchip_plastic_connections(matrix_learning)

bioamp.map_bioamp_reservoir_broadcast_learning(n_columns=3)
nsetup.mapper._program_detail_mapping(0)

chip.load_parameters('biases/biases_reservoir_synthetic_stimuli.biases')

print "now load bioamp biases, when done press input"
variable = raw_input('when done press ENTER')

#present all audio signal and record it
print "recording audio signal.... 15 seconds"
nsetup.mapper._program_detail_mapping(0)
command1 = subprocess.Popen(['sh', '/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/do_stim.sh'])
out = nsetup.stimulate({},send_reset_event=True,duration=15000)
signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)


print "now select which trace to teach, input the time of beginning"
teacher_start = raw_input('when done press ENTER')
teacher_start = int(teacher_start)

#print "we select a single burst and we add the teaching signal"
#ff = out[0].firing_rate(100)
#maxs_ff, mins_ff = functions.peakdet(ff[300], 1000)
#maxs_ff = np.array(maxs_ff)

stimulus_spikes = out[0].raw_data()
index_up = np.where(stimulus_spikes[:,1] == 300)
index_dn = np.where(stimulus_spikes[:,1] == 305)

this_syn_index = rcnpop.synapses['virtual_exc'].addr
# ones we have the spiketimes per cell we have to multiplex them because
# SpikeList accepts a long list of (id, spiketime)...
sl = r_[[zip(repeat(a, len(s)), s)\
    for a, s in zip(broadcast_pop[range(256)].synapses['broadcast'][1::256][0:3].laddr, stimulus_spikes[index_up,0]-np.min(stimulus_spikes[index_up,0]))]].reshape(-1, 2)
sl_1 = r_[[zip(repeat(a, len(s)), s)\
    for a, s in zip(broadcast_pop[range(256)].synapses['broadcast'][1::256][4:7].laddr, stimulus_spikes[index_dn,0])]].reshape(-1, 2)    
id_list = broadcast_pop[range(256)].synapses['broadcast'][1::256].laddr
stim = SpikeList(sl, id_list=id_list)
stim_1 = SpikeList(sl_1, id_list=id_list)

stimulus_ud = pyNCS.pyST.merge_sequencers({1:stim}, {1:stim_1})

def _ismember(a, b):
    '''
    as matlab: ismember
    '''
    tf = np.array([i in b for i in a])
    u = np.unique(a[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
    return tf, index
    
#add teaching signal
#variable = raw_input('when done press ENTER: FILENAME biases/bioamp_go_15gen2015_va.txt')
nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_tbiocas15.biases')
chip.configurator._set_neuron_tau1([255])
tf,index_syn = _ismember(rcnpop.synapses['virtual_exc'].addr['neu'],np.linspace(0,5,6))
syn_teacher = rcnpop.synapses['virtual_exc'][tf]
teacher_signal = syn_teacher.spiketrains_poisson(500,t_start=teacher_start,duration=500)
stimulus_final = pyNCS.pyST.merge_sequencers(teacher_signal, stimulus_ud)
out_chip_teaching = nsetup.stimulate(stimulus_final,send_reset_event=False, duration=15000+800)

pyNCS.monitors.RasterPlot([out_perc,out_rcn])

tf,index_syn = _ismember(rcnpop.synapses['virtual_exc'].addr['neu'],np.linspace(0,5,6))
syn_teacher = rcnpop.synapses['virtual_exc'][tf]
test_signal = syn_teacher.spiketrains_poisson(28,t_start=teacher_start,duration=500)
stimulus_test = pyNCS.pyST.merge_sequencers(test_signal, stimulus_ud)

#chip.load_parameters('biases/biases_reservoir_synthetic_stimuli.biases')
#nsetup.chips['mn256r1'].load_parameters('biases/biases_learning_bioamp.biases')
#variable = raw_input('when done press ENTER: FILENAME biases/bioamp_go_15gen2015_va.txt')
#chip.configurator.set_parameter("VA_EXC_N", 9e-6) # set teached biases
nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_tbiocas15.biases')
chip.configurator._set_neuron_tau1([255])
out_chip = nsetup.stimulate(stimulus_test,send_reset_event=False, duration=15000+800)                     
pyNCS.monitors.RasterPlot([out_perc,out_rcn])
      
       
       
