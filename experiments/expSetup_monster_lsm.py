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
sys.path.append('../api/lsm/')
sys.path.append('../api/retina/')
sys.path.append('../gui/reservoir_display/')
import lsm as L
import time
import retina as ret

######################################
# Configure chip
try:
  is_configured
except NameError:
  print "Configuring chip"
  is_configured = False
else:
  print "Chip is configured: ", is_configured

save_data_to_disk = False
use_retina = True

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
    liquid = L.Lsm(rcnpop, cee=0.4, cii=0.3)

    #c = 0.2
    #dim = np.round(np.sqrt(len(liquid.rcn.synapses['virtual_exc'].addr)*c))

    if(use_retina):
        use_retina_projections = (liquid.matrix_programmable_rec != 1)
        liquid.matrix_programmable_exc_inh[use_retina_projections] = np.random.choice([0,1], size=np.sum(use_retina_projections))
        liquid.matrix_programmable_w[use_retina_projections] = np.random.choice([0,1,2,3], size=np.sum(use_retina_projections))
        liquid.program_config()
        ###### configure retina
        inputpop = pyNCS.Population('','')
        inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  
        syncpop = pyNCS.Population("","")
        syncpop.populate_by_id(nsetup,'mn256r1', 'excitatory', [nsync])
        #reset multiplexer
        chip.configurator._set_multiplexer(0)
        retina = ret.Retina(inputpop)
        retina._init_fpga_mapper()
        retina.map_retina_sync(syncpop, ncol_retina = ncol_retina_sync, neu_sync=nsync) ### neuron 255 is our sync neuron
        rr, pre, post  = retina.map_retina_random_connectivity(inputpop, c = 0.4, syntype='virtual_exc', ncol_sync = ncol_retina_sync)
        rr, pre, post  = retina.map_retina_random_connectivity(inputpop, c = 0.1, syntype='virtual_inh', ncol_sync = ncol_retina_sync)

        import RetinaInputs as ri
        win = ri.RetinaInputs(nsetup)
        #win.run(300)
    
    if use_retina:
        chip.load_parameters('biases/biases_reservoir_retina.biases')
    else:
        chip.load_parameters('biases/biases_reservoir.biases')
    
    # do config only once
    is_configured = True
  
  
# Number of time steps to sample the mean rates
nsteps = 50
######################################

# Stimulation parameters
duration   = 3000   #ms
delay_sync = 500
framerate = 60
counts = (duration/1000)*framerate #for 60 frames per sec
factor_input = 10 # 100
n_neu = 256

# Time vector for analog signals
Fs    = 100/1e3 # Sampling frequency (in kHz)
T     = duration+delay_sync+1000
nT    = np.round (Fs*T)
timev = np.linspace(0,T,nT)

#Conversion from spikes to analog
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*100**2)))
# Function to calculate region of activity
func_avg = lambda t,ts: np.exp((-(t-ts)**2)/(2*150**2)) # time in ms


### TRAINIG SET and test set (as subsample)
theta = linspace(0,pi/2.0,16);
phi  = linspace(90,120,8)*pi/180.0;
[T,P] = np.meshgrid(theta,phi);
wX = (np.sin(P)*np.cos(T)).ravel() 
wY = (np.sin(P)*np.sin(T)).ravel() 
wZ = np.cos(P).ravel()
n_teach = np.prod(np.shape(wX))        
        
for ind in range(n_teach):


    omegas = [wX[ind],wY[ind],wZ[ind]]      
    inputs, output, time_exc = win.run(nT, framerate=framerate, neu_sync = nsync, omegas = omegas)    
    
    for this_t in xrange(ntrials): 

        inputs, outputs, time_exc  = win.run(nT, framerate=framerate, neu_sync = nsync, omegas = omegas)        
        #from 6x3 input signals to matrix of nTx256 signals
        X = np.zeros([nT,n_neu])
        a,b,c = np.shape(win.inputs_signals)
        X[:,0:b*c] = np.reshape(win.inputs_signals, [nT,b*c])
                                
        if save_data_to_disk:
            print "save data"        
            np.savetxt("lsm_ret/inputs_gesture_"+str(ind)+"_trial_"+str(this_t)+".txt", inputs[0])
            np.savetxt("lsm_ret/outputs_gesture_"+str(ind)+"_trial_"+str(this_t)+".txt", outputs[0])
            for i in range(b*c):
                np.savetxt("lsm_ret/teaching_signals_gesture_"+str(ind)+"_teach_input_"+str(i)+"_trial_"+str(this_t)+".txt", X[:,i])



def test_determinism(ntrials):
    for i in range(ntrials):
        inputs, outputs, time_exc = win.run(nT, framerate=framerate, neu_sync = nsync)
        Y = L.ts2sig(timev, membrane, outputs[0][:,0], outputs[0][:,1], n_neu = 256)
        print "we are plotting outputs"
        figure(fig_h.number)
        for i in range(255):
            subplot(16,16,i)
            plot(Y[:,i])
            axis('off')







