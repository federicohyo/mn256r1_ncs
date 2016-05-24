# === Attractors with  MN256R1 =============================================
# === federico@ini.phys.ethz.ch
# ===============================================================================

###################
# Import libraries
###################
import pyNCS
import os as os
import math
import numpy as np
from pylab import *
import time
import sys
# no need to change these library files
sys.path.append('_utils/')
sys.path.append('_utils/mapperlib')
sys.path.append('_utils/biasgenlib')

# here you need to confiture your mapper
sys.path.append('mapper/') 

import biasusb_wrap
import mapper_wrap
import mapper
import AERmn256r1
import MN256R1Configurator
import functions

failed = False

#init mappings
offsetfirstchip = 500
offset_usb = 2000
offset_adcs = 2600
print "Init mapper ..."
mapper.init_mappings(offsetfirstchip,offset_usb,offset_adcs)

#we set the default biases reset latches and so on
biasconf = MN256R1Configurator.MN256R1Configurator()
default_biases = biasconf.load_default(def_file='biases/3_attractors_wta.txt')
biasconf.set_all_biases(default_biases)

### remove neuron 256 from available ones
import conf_mn256r1
conf_mn256r1.available_neus.pop() #it is our clock neuron
AERmn256r1.set_neuron_tau_1([255])

################################
## set up the network
################################
execfile('do_net.py')

################################
## program the network on chip
################################
execfile('program_net.py')

################################
# pyNCS populations
################################
import pyNCS
from pyNCS.neurosetup import NeuroSetup
import sys


# Define file paths
prefix='./chipfiles/'
setuptype = 'setupfiles/mc_final_mn256r1.xml'
setupfile = 'setupfiles/final_mn256r1.xml'

#Create Chip objects
setup = pyNCS.NeuroSetup(setuptype,setupfile,prefix=prefix)
chip = setup.chips['mn256r1']

#chip population
mypop_e = [pyNCS.Population('neurons', 'for fun') for i in range(len(popsne))]
[mypop_e[i].populate_by_id(setup,'mn256r1','excitatory',np.array(popsne[i])) for i in range(len(popsne))]

#chip population
mypop_i = [pyNCS.Population('neurons', 'for fun') for i in range(len(popsne))]
[mypop_i[i].populate_by_id(setup,'mn256r1','excitatory',np.array(popsni[i])) for i in range(len(popsni))]

#all neurons
noise_in = pyNCS.Population('neurons', 'for fun')
noise_in.populate_by_id(setup,'mn256r1','excitatory',np.linspace(0,255,256))

#reading programmable synapses matrix
nsyn_programmable = conf_mn256r1.matrix_programmable_exc_inh[popsne[i],:]

mon_pops_ne = [pyNCS.monitors.SpikeMonitor(mypop_e[i].soma) for i in range(len(popsne))]
mon_pops_ni = [pyNCS.monitors.SpikeMonitor(mypop_i[i].soma) for i in range(len(popsni))]

setup.monitors.import_monitors([ mon_pops_ne[i] for i in range(len(popsne)) ])
setup.monitors.import_monitors([ mon_pops_ni[i] for i in range(len(popsni)) ]) 


#####################################
# define stimulus functions
#####################################
def create_stim_pop_e_virtual(popsne,mypop_e,id_pop,freq,duration, nsteps=5, t_start=5):

    timebins = np.linspace(t_start,duration+t_start,nsteps)
    n_neu = len(popsne[id_pop])
    stim_fixed = r_[[np.linspace(freq,freq,nsteps)]*len(popsne[id_pop])*1]
    stim_matrix = stim_fixed
    tf,index = functions.ismember(mypop_e[id_pop].soma.laddr,popsne[id_pop])
    virtual_syn = mypop_e[id_pop][tf].synapses['virtual_exc'][3::4]
    spiketrain_pop_a = virtual_syn.spiketrains_inh_poisson(stim_matrix,timebins)

    return spiketrain_pop_a


######################################
# run the experiment
######################################
#create stimulus for excitatory populations
duration = 1000
freq_stim = 500  
syn_ = mypop_e[0].synapses['virtual_exc'][3::4]
spike_train_tot = syn_.spiketrains_regular(0.1,10)
ss= 0
for i in range(len(popsne)):
    spiketrain_pop_a = create_stim_pop_e_virtual(popsne,mypop_e,i,freq_stim,duration, nsteps=5, t_start=ss)
    spike_train_tot = pyNCS.pyST.merge_sequencers(spiketrain_pop_a,spike_train_tot)
    ss = duration+ss+1000
for i in range(len(popsne)):
    spiketrain_pop_a = create_stim_pop_e_virtual(popsni,mypop_i,i,freq_stim,duration, nsteps=5, t_start=ss)
    spike_train_tot = pyNCS.pyST.merge_sequencers(spiketrain_pop_a,spike_train_tot)
    ss = duration+ss+1000
    print ss


setup.stimulate(spike_train_tot, send_reset_event=False, duration=ss+8000)    

x_start = np.min(mon_pops_ne[0].sl.raw_data()[:,0])
x_stop = np.max(mon_pops_ne[0].sl.raw_data()[:,0])
pyNCS.monitors.MeanRatePlot(mon_pops_ne)
xlim([x_start-500,x_stop+8500])


########################################
# you can now play with the attractors
########################################
def kill_all():
    duration = 1000
    freq_stim = 500  
    for i in range(len(popsne)):
        spiketrain_pop_inh = create_stim_pop_e_virtual(popsni,mypop_i,i,freq_stim,duration, nsteps=5)
        setup.stimulate(spiketrain_pop_inh, send_reset_event=False, tDuration=duration+3000)    
    time.sleep(1)     


def stimulate_seq():
    #create stimulus for excitatory populations
    duration = 1000
    freq_stim = 500  
    for i in range(len(popsne)):
        spiketrain_pop_a = create_stim_pop_e_virtual(popsne,mypop_e,i,freq_stim,duration, nsteps=5)
        setup.stimulate(spiketrain_pop_a, send_reset_event=False, tDuration=duration+3000)    

        time.sleep(1) 


print '##############################################################'
print '##### you can now play with the attractors ###################'
print '##############################################################'
print '### function availables: kill_all() and stimulate_seq() ######'




