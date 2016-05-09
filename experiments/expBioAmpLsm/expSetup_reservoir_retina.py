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

save_data_to_disk = True
use_retina = True
run_trial = True

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
    liquid = L.Lsm(rcnpop, cee=0.55, cii=0.25)

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
        rr, pre_exc, post_exc  = retina.map_retina_random_connectivity(inputpop, c = 0.4, syntype='virtual_exc', ncol_sync = ncol_retina_sync)
        rr, pre_inh, post_inh  = retina.map_retina_random_connectivity(inputpop, c = 0.1, syntype='virtual_inh', ncol_sync = ncol_retina_sync)

        import RetinaInputsNewSimple as ri
        win = ri.RetinaInputsNewSimple(nsetup)
        #win.run(300)
    
    chip.load_parameters('biases/biases_reservoir_retina2.biases')

    # do config only once
    is_configured = True
 
######################################  
# Number of time steps to sample the mean rates
nsteps = 50

# Stimulation parameters
duration   = 4000   #ms
delay_sync = 500
framerate = 25
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
     
if save_data_to_disk:
    np.savetxt("lsm_ret/theta.txt", theta)
    np.savetxt("lsm_ret/phi.txt", phi)
    np.savetxt("lsm_ret/timev.txt", timev)
    
ntrials = 2 #repeat each point n times    
speed = 2.5
    
if(run_trial):        
    for ind in range(n_teach):
        print '############## gesture ', ind , ' of ', n_teach

        omegas = [wX[ind],wY[ind],wZ[ind]]      
        
        for this_t in xrange(ntrials): 
            print '~~~~~~ trial ', this_t , ' of ', ntrials 
            inputs, outputs, time_exc  = win.run(duration, framerate=framerate, neu_sync = nsync, axi = omegas, speed= speed)        
           
            if save_data_to_disk:
                print "save data"        
                np.savetxt("lsm_ret/inputs_gesture_"+str(ind)+"_trial_"+str(this_t)+".txt", inputs[0])
                np.savetxt("lsm_ret/outputs_gesture_"+str(ind)+"_trial_"+str(this_t)+".txt", outputs[0])
                np.savetxt("lsm_ret/omegas_gesture_"+str(ind)+"_trial_"+str(this_t)+".txt", omegas)

def test_determinism(ntrials):
    ion()
    fig_h = figure()
    for i in range(ntrials):
        inputs, outputs, time_exc = win.run(duration, framerate=framerate, neu_sync = nsync, speed=speed)
        Y = L.ts2sig(timev, membrane, outputs[0][:,0], outputs[0][:,1], n_neu = 256)
        print "we are plotting outputs"
        figure(fig_h.number)
        for i in range(255):
            subplot(16,16,i)
            plot(Y[:,i])
            axis('off')
            
            
## convert retina inputs
def convert_input(inputs):
    '''
    convert 128*128 to 256 ids with macro pixels
    '''
    retina_pixels = 2**14
    num_square = 256
    pixels_per_square = np.floor(retina_pixels/num_square)
    index_list = []
    retina = np.zeros([128,128])
    final_inputs = np.copy(inputs)
    for i in range(16):
        for j in range(16):
            this_square = 0.1*i+0.33*j+0.0537
            index_list.append(this_square)
            retina[j*int(np.sqrt(pixels_per_square)):(j+1)*int(np.sqrt(pixels_per_square)),i*int(np.sqrt(pixels_per_square)):(i+1)*int(np.sqrt(pixels_per_square))] = this_square
    for i in range(256):
        x , y = np.where(retina == index_list[i])    
        indexes = np.ravel_multi_index([x,y], dims=(128,128))
        if(len(indexes) > 0):
            tf = np.in1d(inputs[0][:,1],indexes)#tf,un = _ismember(inputs[0][:,1],indexes)
            final_inputs[0][tf,1] = i
            
    return final_inputs
    
def plot_inputs_small(inputs):                
    ret_image = np.zeros([16,16])
    ret_image = ret_image.flatten()

    for i in range(len(inputs[0][:,0])):
        ret_image[int(inputs[0][i,1])] += 1 
        
    imshow(np.fliplr(ret_image.reshape(16,16).T), interpolation='nearest')
    colorbar()
    

#chip.load_parameters('biases/biases_reservoir_retina2.biases')
def plot_svd():
    inputs, outputs, time_exc = win.run(duration, framerate=framerate, neu_sync = nsync, speed=speed)
    Y = L.ts2sig(timev, membrane, outputs[0][:,0], outputs[0][:,1], n_neu = 256)
    small_inputs = convert_input(inputs)
    X = L.ts2sig(timev, membrane, small_inputs[0][:,0], small_inputs[0][:,1], n_neu = 256)
    figure()
    plot_inputs_small(small_inputs)
    figure()
    plot_inputs_small(outputs)
    figure()
    ac=np.mean(Y**2,axis=0)
    aci=np.mean(X**2,axis=0)
    max_pos = np.where(ac == np.max(ac))[0]
    max_posi = np.where(aci == np.max(aci))[0]
    subplot(3,1,1)
    plot(X[:,max_posi])
    xlabel('time (ds)')
    ylabel('freq (Hz)')
    subplot(3,1,2)
    plot(Y[:,max_pos])
    xlabel('time (ds)')
    ylabel('freq (Hz)')
    subplot(3,1,3)
    CO = np.dot(Y.T,Y)
    CI = np.dot(X.T,X)
    si = np.linalg.svd(CI, full_matrices=True, compute_uv=False)
    so = np.linalg.svd(CO, full_matrices=True, compute_uv=False)
    semilogy(so/so[0], 'bo-', label="outputs")
    semilogy(si/si[0], 'go-', label="inputs")
    xlabel('Singular Value number')
    ylabel('value')
    legend(loc="best") 

