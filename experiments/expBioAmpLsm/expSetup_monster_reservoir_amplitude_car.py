'''
 Copyright (C) 2014 - Federico Corradi
 Copyright (C) 2014 - Juan Pablo Carbajal
 
 This progrm is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

############### author ##########
# federico corradi
# federico@ini.phys.ethz.ch
# Juan Pablo Carbajal 
# ajuanpi+dev@gmail.com
#
# ===============================
#!/usr/env/python
from __future__ import division

import numpy as np
from pylab import *
import pyNCS
import sys
import matplotlib
sys.path.append('../api/reservoir/')
sys.path.append('../api/retina/')
import reservoir as L
import time

######################################
# Configure chip
try:
  is_configured
except NameError:
  print "Configuring chip"
  is_configured = False
else:
  print "Chip is configured: ", is_configured

save_data_to_disk = True
use_retina = True
directory = "lsm_amplitude/"

if (is_configured == False):

    #populations divisible by 2 for encoders
    neuron_ids = np.linspace(0,254,255)
    nsync = 255
    ncol_retina_sync = 5
    npops      = len(neuron_ids)

    #setup
    prefix    = '../'
    setuptype = '../setupfiles/mc_final_mn256r1.xml'
    setupfile = '../setupfiles/final_mn256r1_retina_monster.xml'
    nsetup    = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
    nsetup.mapper._init_fpga_mapper()

    chip      = nsetup.chips['mn256r1']
    chip.configurator._set_multiplexer(0)

    #populate neurons
    rcnpop = pyNCS.Population('neurons', 'for fun') 
    rcnpop.populate_by_id(nsetup,'mn256r1','excitatory', neuron_ids)

    #init liquid state machine
    print "################# init onchip recurrent connections... [reservoir] "
    res = L.Reservoir(rcnpop, cee=0.6, cii=0.35)

    #c = 0.2
    #dim = np.round(np.sqrt(len(liquid.rcn.synapses['virtual_exc'].addr)*c))
    chip.load_parameters('biases/biases_reservoir_amplitude.biases')

    # do config only once
    is_configured = True
 
c = 0.013
duration = 250
delay_sync = 550
max_freq = 385
min_freq = 5 
nsteps = 25 
freq_sync = 1000
duration_sync = 500
ntrials = 1 
steps = np.linspace(0,1,nsteps)
index_syn = None

Fs    = 100/1e3                 # Sampling frequency (in kHz)
T     = duration+delay_sync
nT    = np.round (Fs*T)         # Number of time steps to sample the mean rates
timev = np.linspace(0,T,nT)
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*8**2)))
func_avg = lambda t,ts: np.exp((-(t-ts)**2)/(2*8**2)) # Function to calculate region of activity
    
def test_determinism():
    #inputs, outputs = res.stimulate_reservoir(stimulus, neu_sync = 10,  trials=ntrials)     
    Y = L.ts2sig(timev, membrane, outputs[0][:,0], outputs[0][:,1], n_neu = 256)

    figure(fid_h.number)
    for i in range(256):
        subplot(16,16,i)
        plot(Y[:,i])
        axis('off') 

fid_h = figure()
for index,this_step in enumerate(steps):

    M = np.zeros([2,300])+this_step    
    stimulus, index_syn = res.create_spiketrain_from_amplitude(M, c = c , duration=duration, delay_sync=delay_sync, freq_sync=freq_sync, max_freq= max_freq, min_freq = min_freq, index_syn=index_syn)  
                                                                
    inputs, outputs = res.stimulate_reservoir(stimulus, neu_sync = 10,  trials=ntrials)     

    np.savetxt(directory+"inputs_amplitude_"+str(this_step)+"_trial_"+str(index)+".txt", inputs[0])
    np.savetxt(directory+"outputs_amplitude_"+str(this_step)+"_trial_"+str(index)+".txt", outputs[0])
#    np.savetxt(directory+"index_syn_"+str(this_step)+"_trial_"+str(index)+".txt", index_syn)

#save network configuration    
np.savetxt(directory+"index_projections.txt", index_syn)
res.save_config(directory)

