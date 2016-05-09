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

ion()


prefix='../'  
setuptype = '../setupfiles/mc_final_mn256r1_adcs.xml'
setupfile = '../setupfiles/final_mn256r1_adcs.xml'
nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
chip = nsetup.chips['mn256r1']

p = pyNCS.Population('', '')
p.populate_all(nsetup, 'mn256r1', 'excitatory')

inputpop = pyNCS.Population('','')
inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  
broadcast_pop = pyNCS.Population('','')
broadcast_pop.populate_all(nsetup,'mn256r1', 'excitatory')  

#reset multiplexer
bioamp = Bioamp(inputpop)
bioamp._init_fpga_mapper()
#map up,dn channels to chip
bioamp.map_bioamp_reservoir_broadcast(n_columns=3)
nsetup.mapper._program_detail_mapping(0)    

#chip.load_parameters('biases/biases_reservoir_synthetic_stimuli.biases')
#then load bioamp biases on top of these
rcnpop = pyNCS.Population('neurons', 'for fun') 
rcnpop.populate_all(nsetup,'mn256r1','excitatory')
res = L.Reservoir() #offline build

####configure chip matrixes
print "configure chip matrixes"
matrix_b = np.random.choice([0,0,1],[256,256])
matrix_e_i = np.random.choice([0,0,1,1,1],[256,256])
index_w_1 = np.where(matrix_b == 1)
matrix_weight = np.zeros([256,256])
matrix_weight[index_w_1] = 1
index_w_2 = np.where(matrix_weight != 1)
matrix_weight[index_w_2] = 2

matrix_recurrent = np.random.choice([0,0,1,1,1],[256,256])
matrix_recurrent[index_w_1] = 0

nsetup.mapper._program_onchip_programmable_connections(matrix_recurrent)
nsetup.mapper._program_onchip_broadcast_programmable(matrix_b)
nsetup.mapper._program_onchip_weight_matrix_programmable(matrix_weight) #broadcast goes to weight 1 the rest is w 2
nsetup.mapper._program_onchip_exc_inh(matrix_e_i)
chip.load_parameters('biases/biases_reservoir_synthetic_stimuli.biases')

print "now load bioamp biases, when done press input"
variable = raw_input('when done press ENTER')

import wave
import sys
spf = wave.open('/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/Hackenelektrodenableitung_mecopoda_elongata_chirper_2.wav','r')
#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal_cricket = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
time_orig_cricket=np.linspace(0, len(signal_cricket)/fs, num=len(signal_cricket))


##############
# PARAMETERS
##############

duration_rec = 3500
delta_mem = 3
# Time vector for analog signals
Fs    = 1000/1e3 # Sampling frequency (in kHz)
T     = duration_rec
nT    = np.round (Fs*T)
timev = np.linspace(0,T,nT)
#Conversion from spikes to analog
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))

#### artificial teaching signal
nScales = 3
n_components = 3       # number frequency components in a single gestures
max_f =        5        # maximum value of frequency component in Hz
min_f =        1       # minimum value of frequency component in Hz
nx_d = 2              # 2d input grid of neurons
ny_d = 2 
nScales = 4     
teach_scale = np.linspace(0.05, 0.7, nScales)   # the scales in parallel
num_gestures  = 1
func_avg = lambda t,ts: np.exp((-(t-ts)**2)/(2*50**2)) # Function to calculate region of activity
# -------------
G, rates, gestures = res.generates_gestures(num_gestures, n_components, max_f = max_f, min_f = min_f, nScales = nScales)
M_tot = np.zeros([nx_d*ny_d, nT, num_gestures])
for ind in range(num_gestures):
    this_g = [G[(ind*n_components)+this_c] for this_c in range(n_components)]
    this_r = [rates[(ind*n_components)+this_c] for this_c in range(n_components)]
    M_tot[:,:,ind] = res.create_stimuli_matrix(this_g, this_r, nT, nx=nx_d, ny=ny_d)
# -------------
teach_sig = res.generate_teacher(gestures[0], rates, n_components, nT, nScales, timev, teach_scale)

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
        
        
#reset previous learning
res.reset()
nsetup.mapper._program_detail_mapping(2**3+2**4)    


figs = figure()
#################
## TEACH
#################
n_neu_scale = [256]
for this_readout in range(len(n_neu_scale)):
    res.Nn = n_neu_scale[this_readout]
    res.reset()
    for this_trial in range(3):
        out = nsetup.stimulate({},send_reset_event=True,duration=200)
        out = nsetup.stimulate({},send_reset_event=False,duration=300)
        out = nsetup.stimulate({},send_reset_event=True,duration=duration_rec)
        signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)
        signal = [signal[:,0]-np.min(signal[:,0]),signal[:,1]]
        signal = np.array(signal)
        signal = signal.transpose()
        #extract input and output
        raw_data = out[0].raw_data()
        dnch = 300
        upch = 305
        index_dn = np.where(raw_data[:,1] == dnch)[0]
        index_up = np.where(raw_data[:,1] == upch)[0]
        raw_data_input = []
        raw_data_input.extend(raw_data[index_dn,:])
        raw_data_input.extend(raw_data[index_up,:])
        raw_data_input = np.reshape(raw_data_input,[len(index_dn)+len(index_up),2])
        index_up = np.where(raw_data_input[:,1] == upch)[0]
        index_dn = np.where(raw_data_input[:,1] == dnch)[0]
        raw_data_input[index_dn,1] = 1
        raw_data_input[index_up,1] = 0   
        raw_data = out[0].raw_data()
        index_to_del = np.where(raw_data[:,1] > 255)
        raw_data = np.delete(raw_data, index_to_del,axis=0)

        #teach on reconstructed signal       
        X = L.ts2sig(timev, membrane, raw_data_input[:,0]-np.min(raw_data_input[:,0]), raw_data_input[:,1], n_neu = n_neu_scale[this_readout])
        index_to_remove = np.where(raw_data[:,1] >= n_neu_scale[this_readout])[0]
        raw_data = np.delete(raw_data, index_to_remove,axis=0)
        Y = L.ts2sig(timev, membrane, raw_data[:,0]-np.min(raw_data[:,0]), raw_data[:,1], n_neu = n_neu_scale[this_readout])     
        #teaching signal is interpolated to match actual len of signals
        signal[:,1] = signal[:,1]/abs(signal[:,1]).max()
        signal_ad = [signal[:,0], signal[:,1]]
        ynew = np.linspace(np.min(signal_ad[0]),np.max(signal_ad[0]),nT)
        s = interpolate.interp1d(signal_ad[0], signal_ad[1],kind="linear")
        teach_sig = s(ynew)  
        tmp_ac = np.mean(func_avg(timev[:,None], raw_data[:,0][None,:]-np.min(raw_data[:,0][None,:])), axis=1) 
        tmp_ac = tmp_ac / np.max(tmp_ac)
        ac = tmp_ac[:,None]
        teach_sig = teach_sig * ac**2 # Windowed by activity
        print "teach_sign", np.shape(teach_sig)

        ### TRAIN
        teac_inputs = X[:,1]/abs(X[:,1]).max()    
        res.train(X,Y,teac_inputs[:,None]) #X[:,1][:,None]
        zh = res.predict(X, Y)
        xnorm = X[:,1][:,None]/abs(X[:,1][:,None]).max()
        zhnorm = zh["output"]/abs(zh["output"]).max()
        figure(figsize=(8,10))
        subplot(4,1,1)
        plot(time_orig_cricket,signal_cricket)
        axis('off')
        subplot(4,1,2)
        plot(out[0].raw_data()[:,0], out[0].raw_data()[:,1],'go', markersize=0.5)
        ylabel('neu id', fontsize=18)
        tick_params(axis='both', which='major', labelsize=14)
        ylim([0,256])
        xlim([0,3500])
        subplot(4,1,3)
        index_non_zeros = []
        for i in range(n_neu_scale[this_readout]):
            if(np.sum(Y[:,i]) != 0):
                index_non_zeros.append(i)  
        size_f = np.floor(np.sqrt(len(index_non_zeros)))
        #ynorm = Y/abs(Y).max()
        for i in range(int(size_f**2)):
            #subplot(size_f,size_f,i) 
            plot(Y[:,index_non_zeros[i]])
            #plot(ynorm[:,index_non_zeros[i]]) 
            #axis('off') 
        xlim([0,3500])
        tick_params(axis='both', which='major', labelsize=14)
        ylabel('activations units', fontsize=18) 
        subplot(4,1,4)
        xnorm = X[:,1][:,None]/abs(X[:,1][:,None]).max()
        plot(timev[::],xnorm,label='teaching signal')
        zhnorm = zh["output"]/abs(zh["output"]).max()
        plot(timev[::],zhnorm, label='decoded')
        ylabel('norm. activity', fontsize=18) 
        xlabel('Time [ms]', fontsize=18)
        tick_params(axis='both', which='major', labelsize=14)
        legend(loc='best')
        

        #figure()
        #vlines(raw_data_input[:,0], raw_data_input[:,1]+0.5, raw_data_input[:,1]+1.5)

        #out[0].raster_plot()
        #ylim([0,256])

        #figure()
        #subplot(4,1,1)
        #plot(signal[:,0], signal[:,1])

        #subplot(4,1,2)
        #plot(X[:,0])
        #plot(X[:,1])

        #subplot(4,1,3)       
        #index_non_zeros = []
        #for i in range(256):
        #    if(np.sum(Y[:,i]) != 0):
        #        index_non_zeros.append(i)  
        #size_f = np.floor(np.sqrt(len(index_non_zeros)))
        #for i in range(int(size_f**2)):
            #subplot(size_f,size_f,i) 
        #    plot(Y[:,index_non_zeros[i]])  
            #axis('off')  
        
        
        #teac_inputs = X[:,1]/abs(X[:,1]).max()    
        #res.train(X,Y,teac_inputs[:,None]) #X[:,1][:,None]
        #zh = res.predict(X, Y)
        #subplot(4,1,4)   

        this_rmse = res.root_mean_square(teac_inputs[:,None], zhnorm)
        print "RMSE testing: ", this_rmse
        print "n neu: ", n_neu_scale[this_readout]

        

    #################
    ## TEST
    #################
    figs = figure()
    out = nsetup.stimulate({},send_reset_event=True,duration=200)
    out = nsetup.stimulate({},send_reset_event=False,duration=300)
    out = nsetup.stimulate({},send_reset_event=False,duration=duration_rec)
    signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)
    teacher_start = raw_input('enter teacher start when done press ENTER')
    teacher_start = int(teacher_start)

    signal = [signal[:,0]-np.min(signal[:,0]),signal[:,1]]
    signal = np.array(signal)
    signal = signal.transpose()
    #extract input and output
    raw_data = out[0].raw_data()
    dnch = 300
    upch = 305
    index_dn = np.where(raw_data[:,1] == dnch)[0]
    index_up = np.where(raw_data[:,1] == upch)[0]
    raw_data_input = []
    raw_data_input.extend(raw_data[index_dn,:])
    raw_data_input.extend(raw_data[index_up,:])
    raw_data_input = np.reshape(raw_data_input,[len(index_dn)+len(index_up),2])
    index_up = np.where(raw_data_input[:,1] == upch)[0]
    index_dn = np.where(raw_data_input[:,1] == dnch)[0]
    raw_data_input[index_dn,1] = 1
    raw_data_input[index_up,1] = 0   
    raw_data = out[0].raw_data()
    index_to_del = np.where(raw_data[:,1] > 255)
    raw_data = np.delete(raw_data, index_to_del,axis=0)

    #teach on reconstructed signal
    X = L.ts2sig(timev, membrane, raw_data_input[:,0]-np.min(raw_data_input[:,0]), raw_data_input[:,1], n_neu = n_neu_scale[this_readout])
    index_to_remove = np.where(raw_data[:,1] >= n_neu_scale[this_readout])[0]
    raw_data = np.delete(raw_data, index_to_remove,axis=0)
    Y = L.ts2sig(timev, membrane, raw_data[:,0]-np.min(raw_data[:,0]), raw_data[:,1], n_neu = n_neu_scale[this_readout])     
    #teaching signal is interpolated to match actual len of signals
    zh = res.predict(X, Y)

    #figure()
    #title('PREDICT')
    #subplot(2,1,1)
    #plot(signal[:,0], signal[:,1])

    #subplot(2,1,2)   
    #xnorm = X[:,1][:,None]/abs(X[:,1][:,None]).max()
    #plot(timev[::],xnorm,label='target')
    #zhnorm = zh["output"]/abs(zh["output"]).max()
    #plot(timev[::],zhnorm, label='predicted')
    #legend(loc='best')   
                
    figure(figsize=(8,10))
    subplot(4,1,1)
    plot(time_orig_cricket,signal_cricket)
    axis('off')
    subplot(4,1,2)
    plot(out[0].raw_data()[:,0], out[0].raw_data()[:,1],'go', markersize=0.5)
    ylabel('neu id', fontsize=18)
    tick_params(axis='both', which='major', labelsize=14)
    ylim([0,256])
    xlim([0,3500])
    subplot(4,1,3)
    index_non_zeros = []
    for i in range(n_neu_scale[this_readout]):
        if(np.sum(Y[:,i]) != 0):
            index_non_zeros.append(i)  
    size_f = np.floor(np.sqrt(len(index_non_zeros)))
    #ynorm = Y/abs(Y).max()
    for i in range(int(size_f**2)):
        #subplot(size_f,size_f,i) 
        plot(Y[:,index_non_zeros[i]])
        #plot(ynorm[:,index_non_zeros[i]]) 
        #axis('off') 
    xlim([0,3500])
    tick_params(axis='both', which='major', labelsize=14)
    ylabel('activations units', fontsize=18) 
    subplot(4,1,4)
    xnorm = X[:,1][:,None]/abs(X[:,1][:,None]).max()
    plot(timev[::],xnorm,label='target signal')
    zhnorm = zh["output"]/abs(zh["output"]).max()
    plot(timev[::],zhnorm, label='decoded')
    ylabel('norm. activity', fontsize=18) 
    xlabel('Time [ms]', fontsize=18)
    tick_params(axis='both', which='major', labelsize=14)
    legend(loc='best')

    this_rmse = res.root_mean_square(teac_inputs[:,None], zhnorm)
    print "RMSE testing: ", this_rmse
    print "n neu: ", n_neu_scale[this_readout]
    print "#############################"
              
def plot_svd(X, Y):
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
            
            
            
            
