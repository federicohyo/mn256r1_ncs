'''
 Copyright (C) 2014 - Federico Corradi
 
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
#################################
#!/usr/env/python
from __future__ import division

import numpy as np
from pylab import *
import pyNCS
import sys
import matplotlib
sys.path.append('../api/reservoir/')
sys.path.append('../api/retina/')
sys.path.append('../gui/reservoir_display/')
import reservoir as L
import time
import retina as ret

######################################
# Configure chip
######################################
try:
  is_configured
except NameError:
  print "Configuring chip"
  is_configured = False
else:
  print "Chip is configured: ", is_configured

######################################
# OPTIONS
######################################
save_data_to_disk = True 
real_time_learn = False 
produce_learning_plot = False
test_reservoir = False
teach_orth = False

if (is_configured == False):

    #populations divisible by 2 for encoders
    neuron_ids = np.linspace(0,254,255)
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
    res = L.Reservoir(rcnpop, cee=1.0, cii=0.45)
    res.program_config()    
    #c = 0.2
    #dim = np.round(np.sqrt(len(liquid.rcn.synapses['virtual_exc'].addr)*c))
    chip.load_parameters('biases/biases_reservoir_synthetic_stimuli.biases')
    # do config only once
    is_configured = True
  
######################################
# End chip configuration
######################################

######################################
# Gestures and network parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CUT
######################################
#  ~~~~~~ TRAIN
######################################
num_gestures = 1        # number of gestures
repeat_same  = 70       # n trial
regenerate_input_train = False  #variability on the inputs? resample the input poisson trains
n_components = 3       # number frequency components in a single gestures
max_f =        5        # maximum value of frequency component in Hz
min_f =        1       # minimum value of frequency component in Hz
nx_d = 2              # 2d input grid of neurons
ny_d = 2 
nScales = 4             # number of scales (parallel teaching on different scales, freqs)
teach_scale = np.linspace(0.05, 0.7, nScales)   # the scales in parallel

######################################
#  ~~~~~~ TEST
######################################
num_gestures_test = 1           # number of test gestures 
perturbe_test = True         # perturb a previously learned gestures?
n_perturbations_points = 300             # n*3 is the total number in all direction on an X
n_repeat_perturbations = 1 
center_pert_value_x = np.append(np.linspace(-1, 1,  n_perturbations_points),np.linspace(-1, 1,  n_perturbations_points))    
center_pert_value_y = np.append(np.linspace(1, -1,  n_perturbations_points),np.linspace(-1, 1,  n_perturbations_points))    
randomize_pert = True
# other parameters 
initNt = 0                      # Time to init the reservoir, not training in this initTime 
duration   = 2500   #ms
delay_sync = 500
c = 0.075#0.35  #connectivity with input signals (new parameter in respect to simulation)
Fs    = 100/1e3                 # Sampling frequency (in kHz)
T     = duration+delay_sync
nT    = np.round (Fs*T)         # Number of time steps to sample the mean rates
timev = np.linspace(0,T,nT)
max_freq = 235 #370 poiss #250 reg 
min_freq = 40 

#################
# PLOT
#################
plot_teaching = False
neutoplot = 155         #pick one neuron and we will plot it
plot_all_test = False

#################
# SAVE DATA
#################
directory = 'lsm_synt/'

if save_data_to_disk:
    #TRAIN
    np.savetxt(directory+'num_gestures', [num_gestures])
    np.savetxt(directory+'repeat_same', [repeat_same])    # n trial
    np.savetxt(directory+'n_components', [n_components]) # number frequency components in a single gestures
    np.savetxt(directory+'max_f', [max_f])# maximum value of frequency component in Hz
    np.savetxt(directory+'min_f', [min_f]) # minimum value of frequency component in Hz
    np.savetxt(directory+'nx_d', [nx_d])# 2d input grid of neurons
    np.savetxt(directory+'ny_d', [ny_d]) 
    np.savetxt(directory+'nScales', [nScales])# number of scales (parallel teaching on different scales, freqs)
    np.savetxt(directory+'teach_scale', teach_scale)# the scales in parallel
    #TEST
    np.savetxt(directory+'num_gestures_test', [num_gestures_test])
    np.savetxt(directory+'perturbe_test', [perturbe_test]) # perturb a previously learned gestures?
    np.savetxt(directory+'n_perturbations_points', [n_perturbations_points])
    np.savetxt(directory+'n_repeat_perturbations', [n_repeat_perturbations])
    np.savetxt(directory+'center_pert_value_x', center_pert_value_x)
    np.savetxt(directory+'center_pert_value_y', center_pert_value_y)
    np.savetxt(directory+'initNt', [initNt])
    np.savetxt(directory+'duration', [duration])
    np.savetxt(directory+'delay_sync', [delay_sync])
    np.savetxt(directory+'c', [c])
    np.savetxt(directory+'Fs', [Fs]) 
    np.savetxt(directory+'T', [T])
    np.savetxt(directory+'nT', [nT])    
    np.savetxt(directory+'timev', timev)               
    np.savetxt(directory+'max_freq', [max_freq])        
    np.savetxt(directory+'min_freq', [min_freq])     
    np.savetxt(directory+'neutoplot', [neutoplot])       

####################################
#Conversion from spikes to analog
###################################
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*35**2)))
func_avg = lambda t,ts: np.exp((-(t-ts)**2)/(2*35**2)) # Function to calculate region of activity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END CUT

#####################################
# Build Stimuli
# -------------
# ~~ TRAIN STIM
G, rates, gestures = res.generates_gestures(num_gestures, n_components, max_f = max_f, min_f = min_f, nScales = nScales)
M_tot = np.zeros([nx_d*ny_d, nT, num_gestures])
for ind in range(num_gestures):
    this_g = [G[(ind*n_components)+this_c] for this_c in range(n_components)]
    this_r = [rates[(ind*n_components)+this_c] for this_c in range(n_components)]
    M_tot[:,:,ind] = res.create_stimuli_matrix(this_g, this_r, nT, nx=nx_d, ny=ny_d)
# -------------

res.reset(alpha=np.logspace (-8,12,100))
##################################################
# Train the Network with all train gestures
##################################################
fig_a = figure()
rmse_tot = np.zeros([nScales, num_gestures, repeat_same])
rmse_tot_input = np.zeros([nScales, num_gestures, repeat_same])
for this_g in range(num_gestures):
    # This should be a for loop.... over all gestures 
    #poke network
    tot_inputs = []
    tot_outputs = []
    index_syn = None
    if(regenerate_input_train):
        print 'teaching variability on the input signal'
        for i in range(repeat_same):
            print 'building input stimulus...'
            stimulus, index_syn = res.create_spiketrain_from_matrix_reg_fast(M_tot[:,:,this_g], 
                                                            c = c, 
                                                            duration=duration,  
                                                            delay_sync=delay_sync,  
                                                            max_freq= max_freq, min_freq = min_freq, index_syn= index_syn)
                
            print 'done'
            inputs, outputs = res.stimulate_reservoir(stimulus, neu_sync = 10,  trials=1)
            tot_inputs.append(inputs)
            tot_outputs.append(outputs)
                
        inputs = tot_inputs
        outputs = tot_outputs   
        inputs = (np.reshape(inputs,[repeat_same]))
        outputs = (np.reshape(outputs,[repeat_same]))
    else:
        print 'no variability on train signal'
        print 'building input stimulus...'
        stimulus, index_syn = res.create_spiketrain_from_matrix_reg_fast(M_tot[:,:,this_g], 
                                                            c = c, 
                                                            duration=duration,  
                                                            delay_sync=delay_sync,  
                                                            max_freq= max_freq, min_freq = min_freq)
        print 'done'
        inputs, outputs = res.stimulate_reservoir(stimulus, neu_sync = 10,  trials=repeat_same)
            
            
    #generate associated teacher signal
    teach_sig = res.generate_teacher(gestures[this_g], rates, n_components, nT, nScales, timev, teach_scale)  
    for this_trial in range(repeat_same):
        if real_time_learn:         
        
            if(this_trial == 0):
                # Calculate activity of current inputs.
                # As of now the reservoir can only give answers during activity
                tmp_ac = np.mean(func_avg(timev[:,None], outputs[this_trial][:,0][None,:]), axis=1) 
                tmp_ac = tmp_ac / np.max(tmp_ac)
                ac = tmp_ac[:,None]
                teach_sig = teach_sig * ac**4 # Windowed by activity
                print "teach_sign", np.shape(teach_sig)
                    
            X = L.ts2sig(timev, membrane, outputs[this_trial][:,0], outputs[this_trial][:,1], n_neu = 256)
            Yt = L.ts2sig(timev, membrane, inputs[this_trial][:,0], inputs[this_trial][:,1], n_neu = 256)
            
            res.train(X,Yt,teach_sig)          
            zh = res.predict(X, Yt, initNt=initNt)
            if plot_teaching:
                figure()
                title('training error')
            print '####### gesture n', this_g, 
            for i in range(nScales):
                this_rmse = res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])
                this_rmse_input = res.root_mean_square(teach_sig[initNt::,i], zh["input"][:,i])
                print '### SCALE n', i, ' RMSE ', this_rmse
                rmse_tot[i, this_g, this_trial] = this_rmse
                rmse_tot_input[i, this_g, this_trial] = this_rmse_input
                if plot_teaching:
                    subplot(nScales,1,i+1)
                    plot(timev[initNt::],teach_sig[initNt::,i],label='teach signal')
                    plot(timev[initNt::],zh["input"][:,i], label='input')
                    plot(timev[initNt::],zh["output"][:,i], label='output')
            if plot_teaching:        
                legend(loc='best')
                figure(fig_a.number)
                plot(X[:,neutoplot],label='train')  
                xlabel('Time')
                ylabel('Freq [Hz]')           

        if save_data_to_disk:
            print 'saving data..'
            np.savetxt(directory+"inputs_gesture_"+str(this_g)+"_trial_"+str(this_trial)+".txt", inputs[this_trial])
            np.savetxt(directory+"outputs_gesture_"+str(this_g)+"_trial_"+str(this_trial)+".txt", outputs[this_trial])
            for this_te in range(nScales):
                np.savetxt(directory+"teaching_signals_gesture_"+str(this_g)+"_teach_input_"+str(this_te)+"_trial_"+str(this_trial)+".txt", teach_sig[:,this_te])


##################################################
# BUILD PERTURBED STIMs
##################################################
# ~~ TEST STIM
gestures_final= []
if perturbe_test:
    print 'we perturb teached gestures'
    #perturbe teached gestures
    gestures_pert = (np.repeat(gestures, len(center_pert_value_x))).copy()
    
    for this_test in range(len(center_pert_value_x)): #loop over gestures
        freqs   = gestures_pert[this_test]["freq"]  # in Hz
        width   = gestures_pert[this_test]["width"] 
        centers = []
        for this_component in range(n_components):  
            if(randomize_pert):              
                centers.append([gestures_pert[this_test]["centers"][this_component][0]+ center_pert_value_x[this_test]*np.random.random_sample()- center_pert_value_x[this_test], gestures_pert[this_test]["centers"][this_component][1]+ center_pert_value_y[this_test]*np.random.random_sample()- center_pert_value_y[this_test]]) 
            else:
                centers.append([gestures_pert[this_test]["centers"][this_component][0] + center_pert_value_x[this_test], gestures_pert[this_test]["centers"][this_component][1] + center_pert_value_y[this_test]])    
        gestures_final.append({'freq': freqs, 'centers': centers, 'width': width}) 
      
    G_test, rates_test =  res.generates_G_rates(gestures_final)
    gestures_pert = gestures_final
else:
    #generate new sets of gestures
    G_test, rates_test, gestures_test = res.generates_gestures( num_gestures_test, n_components, max_f = max_f, min_f = min_f, nScales = nScales)
    gestures_pert = gestures_test

M_tot_test = np.zeros([nx_d*ny_d, nT, len(gestures_final)])
for ind in range(len(gestures_final)):
    this_g = [G_test[(ind*n_components)+this_c] for this_c in range(n_components)]
    this_r = [rates_test[(ind*n_components)+this_c] for this_c in range(n_components)]
    M_tot_test[:,:,ind] = res.create_stimuli_matrix(this_g, this_r, nT, nx=nx_d ,ny=ny_d)
 
import json
json.dump(gestures_pert, open(directory+"/gestures_pert.txt",'w'))
json.dump(gestures, open(directory+"/gestures.txt",'w'))
    
##################################################
# TEST the Network with all test gestures
##################################################
rmse_tot_test = np.zeros([nScales, len(gestures_final), n_repeat_perturbations])
for this_g in range(len(gestures_final)):
    stimulus_t, index_syn = res.create_spiketrain_from_matrix_reg_fast(M_tot_test[:,:,this_g], 
                                                c = c, 
                                                duration=duration,  
                                                delay_sync=delay_sync,  
                                                max_freq= max_freq, min_freq = min_freq,
                                                index_syn=index_syn)
    
    inputs, outputs = res.stimulate_reservoir(stimulus_t, neu_sync = 10,  trials=n_repeat_perturbations)
    
    for this_trial in range(n_repeat_perturbations):
        if real_time_learn:        
        
            X = L.ts2sig(timev, membrane, outputs[this_trial][:,0], outputs[this_trial][:,1], n_neu = 256)
            Yt = L.ts2sig(timev, membrane, inputs[this_trial][:,0], inputs[this_trial][:,1], n_neu = 256)
            
            #predict
            zh = res.predict(X, Yt, initNt=initNt)
            
            if(plot_all_test):
                figure()
                title('test error')
            rmse_this_g = []
            for i in range(nScales):
                if(plot_all_test):
                    subplot(nScales,1,i+1)
                    #teach sig is the same used to train
                    plot(timev[initNt::],teach_sig[initNt::,i],label='test target signal')
                    plot(timev[initNt::],zh["input"][:,i], label='input')
                    plot(timev[initNt::],zh["output"][:,i], label='output')
                print "TESTING ERROR RMSE:", res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])
                rmse_this_g = res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])
                rmse_tot_test[i, this_g, this_trial] = rmse_this_g
            if(plot_all_test):    
                legend(loc='best')       
                figure(fig_a.number)
                plot(X[:,neutoplot], 'o-', label='test')
         
        if save_data_to_disk:
            print 'saving data..'
            np.savetxt(directory+"inputs_gesture_perturbed_"+str(this_g)+"_trial_"+str(this_trial)+".txt", inputs[this_trial])
            np.savetxt(directory+"outputs_gesture_perturbed_"+str(this_g)+"_trial_"+str(this_trial)+".txt", outputs[this_trial])
            

    if(plot_all_test):  
        legend(loc='best')


            
########################################################
# USEFUL FUNCTIONS -> DEBUG AND TEST
#######################################################
#fig_h = figure()
#fig_hh = figure()
def test_determinism():
    inputs, outputs, eff_time = win.run(nT, framerate=framerate, neu_sync = nsync, omegas = omegas) 
    Y = L.ts2sig(timev, membrane, outputs[0][:,0], outputs[0][:,1], n_neu = 256)
    #X = L.ts2sig(timev, membrane, inputs[0][:,0], inputs[0][:,1], n_neu = 128*128)  
    X = np.zeros([nT,n_neu])
    a,b,c = np.shape(win.inputs_signals)
    X[:,0:b*c] = np.reshape(win.inputs_signals, [nT,b*c])  

    print "we are plotting outputs"
    figure(fig_h.number)
    for i in range(256):
        subplot(16,16,i)
        plot(Y[:,i])
        axis('off') 
    print "we are plotting svd and inputs/outputs"     
    plot_svd(fig_hh, X, Y)
             
        
def _ismember( a, b):
    '''
    as matlab: ismember
    '''
    # tf = np.in1d(a,b) # for newer versions of numpy
    tf = np.array([i in b for i in a])
    u = np.unique(a[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
    return tf, index        
                
def plot_inputs(inputs):                
    ret_image = np.zeros([128,128])
    ret_image = ret_image.flatten()

    for i in range(len(inputs[0][:,0])):
        ret_image[int(inputs[0][i,1])] += 1 
        
    imshow(np.fliplr(ret_image.reshape(128,128).T))
    colorbar()
        
def plot_inputs_small(inputs):                
    ret_image = np.zeros([16,16])
    ret_image = ret_image.flatten()

    for i in range(len(inputs[0][:,0])):
        ret_image[int(inputs[0][i,1])] += 1 
        
    imshow(np.fliplr(ret_image.reshape(16,16).T), interpolation='nearest')
    colorbar()
    
def plot_svd(fig_hh, X, Y):
    figure(fig_hh.number)
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
        

