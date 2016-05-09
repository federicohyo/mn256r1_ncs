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


directory = 'lsm_synt/'
fig_dir = 'figures/'            #make sure they exists

#TRAIN
num_gestures = int(np.loadtxt(directory+'num_gestures'))
repeat_same = int(np.loadtxt(directory+'repeat_same'))    # n trial
n_components = int(np.loadtxt(directory+'n_components')) # number frequency components in a single gestures
max_f  =  int(np.loadtxt(directory+'max_f'))# maximum value of frequency component in Hz
min_f  = int(np.loadtxt(directory+'min_f')) # minimum value of frequency component in Hz
nx_d   = int(np.loadtxt(directory+'nx_d'))# 2d input grid of neurons
ny_d   = int(np.loadtxt(directory+'ny_d')) 
nScales  =  int(np.loadtxt(directory+'nScales'))# number of scales (parallel teaching on different scales, freqs)
teach_scale =   np.loadtxt(directory+'teach_scale')# the scales in parallel
#TEST
num_gestures_test  =  int(np.loadtxt(directory+'num_gestures_test')) 
perturbe_test  =  int(np.loadtxt(directory+'perturbe_test'))  # perturb a previously learned gestures?
n_perturbations_points =   int(np.loadtxt(directory+'n_perturbations_points')) 
n_repeat_perturbations  =  int(np.loadtxt(directory+'n_repeat_perturbations')) 
center_pert_value_x =   (np.loadtxt(directory+'center_pert_value_x')) 
center_pert_value_y =   (np.loadtxt(directory+'center_pert_value_y')) 
initNt  =  int(np.loadtxt(directory+'initNt')) 
duration = int(np.loadtxt(directory+'duration')) 
delay_sync = int(np.loadtxt(directory+'delay_sync')) 
c   = int(np.loadtxt(directory+'c')) 
Fs  = int(np.loadtxt(directory+'Fs')) 
T   = int(np.loadtxt(directory+'T')) 
nT  = int(np.loadtxt(directory+'nT'))  
timev  = np.loadtxt(directory+'timev')              
max_freq  = int(np.loadtxt(directory+'max_freq'))        #stimulus rescale
min_freq  = int(np.loadtxt(directory+'min_freq'))     
neutoplot = int(np.loadtxt(directory+'neutoplot'))     
     
#load gestures     
import json
with open (directory+"/gestures_pert.txt", "r") as myfile:
    data=myfile.read()
gestures_final = json.loads(data)
with open (directory+"/gestures.txt", "r") as myfile:
    data=myfile.read()
gestures = json.loads(data)

####################################
#Conversion from spikes to analog
###################################
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*35**2)))
func_avg = lambda t,ts: np.exp((-(t-ts)**2)/(2*35**2)) # Function to calculate region of activity
######################## END EXP PARAMETERS

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

######################################
# LOAD DATA and DO EVERYTHING OFFLINE
######################################

res = L.Reservoir() #object without population does not need the chip offline build

fig_a = figure()
fig_all_train = figure()
rmse_tot = np.zeros([nScales, num_gestures, repeat_same])
rmse_tot_input = np.zeros([nScales, num_gestures, repeat_same])

#### TRAIN
for this_g in range(num_gestures):

    teach_sig = np.zeros([nT,nScales])
    for this_te in range(nScales):
        teach_sig[:,this_te] = np.loadtxt(directory+"teaching_signals_gesture_"+str(this_g)+"_teach_input_"+str(this_te)+"_trial_"+str(0)+".txt")
            
                        
    for this_trial in range(repeat_same):
        inputs = np.loadtxt(directory+"inputs_gesture_"+str(this_g)+"_trial_"+str(this_trial)+".txt")
        outputs = np.loadtxt(directory+"outputs_gesture_"+str(this_g)+"_trial_"+str(this_trial)+".txt")
        

        X = L.ts2sig(timev, membrane, outputs[:,0], outputs[:,1], n_neu = 256)
        Yt = L.ts2sig(timev, membrane, inputs[:,0], inputs[:,1], n_neu = 256)  
        
        print "train offline reservoir on gesture..", str(this_g), " trial ", str(this_trial) 
        
        if(this_trial == 0):
            # Calculate activity of current inputs.
            # As of now the reservoir can only give answers during activity
            tmp_ac = np.mean(func_avg(timev[:,None], outputs[:,0][None,:]), axis=1) 
            tmp_ac = tmp_ac / np.max(tmp_ac)
            ac = tmp_ac[:,None]
            teach_sig = teach_sig * ac**4 # Windowed by activity
            print "teach_sign", np.shape(teach_sig)
            
        res.train(X,Yt,teach_sig)   
        zh = res.predict(X, Yt, initNt=initNt)
        

        figure()
        title('training error')
        for i in range(nScales):
            this_rmse = res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])
            this_rmse_input = res.root_mean_square(teach_sig[initNt::,i], zh["input"][:,i])
            print '### SCALE n', i, ' RMSE ', this_rmse
            rmse_tot[i, this_g, this_trial] = this_rmse
            rmse_tot_input[i, this_g, this_trial] = this_rmse_input

            subplot(nScales,1,i+1)
            plot(timev[initNt::],teach_sig[initNt::,i],label='teach signal')
            #plot(timev[initNt::],zh["input"][:,i], label='input')
            plot(timev[initNt::],zh["output"][:,i], label='output')
        savefig(directory+fig_dir+'all_scale_gesture_'+str(this_g)+'_trial_'+str(this_trial)+'_train_.eps', format='eps')
        savefig(directory+fig_dir+'all_scale_gesture_'+str(this_g)+'_trial_'+str(this_trial)+'_train_.png', format='png')

        legend(loc='best')
        figure(fig_a.number)
        plot(X[:,neutoplot],label='train')  
        xlabel('Time')
        ylabel('Freq [Hz]')  
        if(this_trial == repeat_same-1):
            savefig(directory+fig_dir+'neu_num_'+str(neutoplot)+'_trials_train.eps', format='eps')
            savefig(directory+fig_dir+'neu_num_'+str(neutoplot)+'_trials_train.png', format='png')
        
        for this_neu in range(255):
            figure(fig_all_train.number)
            subplot(16,16,this_neu)
            plot(X[:,this_neu],label='train')
            axis('off') 
        if(this_trial == repeat_same-1):
            savefig(directory+fig_dir+'all_neu_all_trials_train.eps', format='eps')
            savefig(directory+fig_dir+'all_neu_all_trials_train.png', format='png')    
        
        
#plot RMSE
figure()
title('outputs training error')
for gesture_t in range(num_gestures):
    for scale_t in range(nScales):
        plot(rmse_tot[scale_t,gesture_t,:], 'o-', label='scale :'+str(teach_scale[scale_t]))
ylabel('RMSE')
xlabel('presentation trial')
legend(loc='best')
savefig(directory+fig_dir+'rmse_train_tot_outputs.eps', format='eps')    
savefig(directory+fig_dir+'rmse_train_tot_outputs.png', format='png')    

figure()
title('inputs training error')
for gesture_t in range(num_gestures):
    for scale_t in range(nScales):
        plot(rmse_tot_input[scale_t,gesture_t,:], 'o-', label='scale :'+str(teach_scale[scale_t]))
ylabel('RMSE')
xlabel('presentation trial')
legend(loc='best')
savefig(directory+fig_dir+'rmse_train_tot_inputs.eps', format='eps')    
savefig(directory+fig_dir+'rmse_train_tot_inputs.png', format='png')    
        
#plot SVD values        
fig_svd = figure()
plot_svd(fig_svd, Yt, X)        
savefig(directory+fig_dir+'svd_input_output_train.eps', format='eps')    
savefig(directory+fig_dir+'svd_input_output_train.png', format='png') 
        
#### TEST DATA
rmse_tot_test = np.zeros([nScales, len(gestures_final), n_repeat_perturbations])
for this_g in range(len(gestures_final)):
     
    for this_trial in range(n_repeat_perturbations):
        inputs = np.loadtxt(directory+"inputs_gesture_perturbed_"+str(this_g)+"_trial_"+str(this_trial)+".txt")
        outputs = np.loadtxt(directory+"outputs_gesture_perturbed_"+str(this_g)+"_trial_"+str(this_trial)+".txt")
       
               
        X = L.ts2sig(timev, membrane, outputs[:,0], outputs[:,1], n_neu = 256)
        Yt = L.ts2sig(timev, membrane, inputs[:,0], inputs[:,1], n_neu = 256)
        
        #predict
        zh = res.predict(X, Yt, initNt=initNt)    

        figure()
        title('test error')
        rmse_this_g = []
        for i in range(nScales):
            subplot(nScales,1,i+1)
            #teach sig is the same used to train
            plot(timev[initNt::],teach_sig[initNt::,i],label='test target signal')
            #plot(timev[initNt::],zh["input"][:,i], label='input')
            plot(timev[initNt::],zh["output"][:,i], label='output')
            print "TESTING ERROR RMSE:", res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])
            rmse_this_g = res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])
            rmse_tot_test[i, this_g, this_trial] = rmse_this_g
        savefig(directory+fig_dir+'all_scale_gesture_'+str(this_g)+'_trial_'+str(this_trial)+'_test_.eps', format='eps')
        savefig(directory+fig_dir+'all_scale_gesture_'+str(this_g)+'_trial_'+str(this_trial)+'_test_.png', format='png')


        legend(loc='best')       
        figure(fig_a.number)
        plot(X[:,neutoplot], 'o-', label='test')
        if(this_trial == n_repeat_perturbations-1):
            savefig(directory+fig_dir+'neu_num_'+str(neutoplot)+'_trials_test_train.eps', format='eps')
            savefig(directory+fig_dir+'neu_num_'+str(neutoplot)+'_trials_test_train.png', format='png')
  
    legend(loc='best')
       
        
figure()
title('output testing perturbed error')
d_t = np.zeros([len(gestures_final), n_components])
for gesture_t in range(len(gestures_final)):
    for scale_t in range(nScales):
        for this_c in range(n_components):
            d_t[gesture_t,this_c] = np.sqrt( (gestures_final[gesture_t]['centers'][this_c][0] - gestures[0]['centers'][this_c][0])**2 + (gestures_final[gesture_t]['centers'][this_c][1] -gestures[0]['centers'][this_c][1])**2 )
        plot(np.repeat(np.mean(d_t[gesture_t,:]),len(rmse_tot_test[scale_t,gesture_t,:])),rmse_tot_test[scale_t,gesture_t,:], 'o-', label='scale :'+str(teach_scale[scale_t]))
ylabel('RMSE')
xlabel('mean perturbation centers')
savefig(directory+fig_dir+'rmse_test_pert_outputs.eps', format='eps')    
savefig(directory+fig_dir+'rmse_test_pert_outputs.png', format='png')    
        
figure()        
cc = np.linspace(0,1,len(gestures_final))
cc_d = 1-np.mean(d_t,axis=1)
coror_error = (rmse_tot_test[0,:,:]/np.max(rmse_tot_test[0,:,:])) ## color are scaled
cm = plt.cm.get_cmap('jet')
scale = 0
x = []
y = []
error_cool = []
for i in range(len(gestures_final)):     
    for this_c in range(n_components):
        x.append(gestures_final[i]['centers'][this_c][0])    
        y.append(gestures_final[i]['centers'][this_c][1])
        error_cool.append(coror_error[i][0])
scatter(x,y,c=error_cool,cmap=cm) 
colorbar()      
for gesture_t in range(num_gestures):
    for this_c in range(n_components):
        scatter(gestures[gesture_t]['centers'][this_c][0], gestures[gesture_t]['centers'][this_c][1], s=12, marker='o', color='r')        
savefig(directory+fig_dir+'rmse_test_pert_outputs_colorscatter.eps', format='eps')    
savefig(directory+fig_dir+'rmse_test_pert_outputs_colorscatter.png', format='png')         
        
        
        
        
