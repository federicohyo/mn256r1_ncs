#!/usr/env/python
import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('/home/federico/projects/work/api/wij/')
sys.path.append('/home/federico/projects/work/api/bioamp/')
sys.path.append('/home/federico/projects/work/api/perceptrons/')
import sys

sys.path.append('/home/federico/projects/work/api/reservoir/')
sys.path.append('/home/federico/projects/work/api/retina/')
from scipy import interpolate
import reservoir as L

from perceptrons import Perceptrons
from wij import SynapsesLearning
from bioamp import Bioamp
import time
import scipy.signal
import subprocess
import scipy.signal.signaltools as sigtool
from os import listdir
from os.path import isfile, join

import pickle
import wave
ion()

use_chip = True 
load_config = True
config_dir = 'data_reservoir/ee03-ii02_b4/'
wavefile =  'Hackenelektrodenableitung_mecopoda_elongata_chirper_2.wav'


if use_chip:
    ##########################################
    # INIT PYNCS and CHIP
    ########################################

    prefix='/home/federico/projects/work/trunk/code/python/mn256r1_ncs/'  
    setuptype = '/home/federico/projects/work/trunk/code/python/mn256r1_ncs/setupfiles/mc_final_mn256r1_adcs.xml'
    setupfile = '/home/federico/projects/work/trunk/code/python/mn256r1_ncs/setupfiles/final_mn256r1_adcs.xml'
    nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
    chip = nsetup.chips['mn256r1']

    p = pyNCS.Population('', '')
    p.populate_all(nsetup, 'mn256r1', 'excitatory')

    inputpop = pyNCS.Population('','')
    inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  

    #reset multiplexer
    bioamp = Bioamp(inputpop)
    bioamp._init_fpga_mapper()


    #######################################################3
    ####  RESERVOIR
    ########################################################

    rcnpop = pyNCS.Population('neurons', 'for fun') 
    rcnpop.populate_all(nsetup,'mn256r1','excitatory')
    res = L.Reservoir(rcnpop, cee=0.4, cii=0.2) #0.8 0.6
    if(load_config == True):
        res.load_config(config_dir)
    res.program_config()    
    res.reset(alpha=np.logspace(-10,100,500))  ##init regressor



    ################################
    ### CONNECT BIOAMP TO RESERVOIR
    ################################

    bioamp._init_fpga_mapper()
    bioamp.map_bioamp_reservoir_broadcast(n_columns=4)
    nsetup.mapper._program_detail_mapping(2**3+2**4)  


    print('load parameters from bioamp interface.. use file biases/bioamp_delta_reservoir.txt or biases/bioamp_delta_reservoir_forest.txt') 
    raw_input("Press Enter to when done...")

    #set programmable synapse parameters
    chip.configurator.set_parameter("NPA_WEIGHT_INH0_N",0.001e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_INH1_N",1.2625e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_INH_N",0.03025e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_EXC_P",1.610925e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_EXC0_P",1.5398e-6)
    chip.configurator.set_parameter("NPA_WEIGHT_EXC1_P",0.5e-6)
    chip.configurator.set_parameter("NPDPIE_THR_P",10e-12)
    chip.configurator.set_parameter("NPDPIE_TAU_P",80e-12)
    chip.configurator.set_parameter("NPDPII_THR_P",200e-12)
    chip.configurator.set_parameter("NPDPII_TAU_P",200e-12)

    #neuron parameters
    chip.configurator.set_parameter("IF_TAU2_N",8.6e-12)
    chip.configurator.set_parameter("IF_TAU1_N",0.033e-6)

    chip.configurator._set_all_neu_tau2()
    taus_1 = np.array([22,20,77,72,100,200,150,80,60,1,5,10,140])
    chip.configurator._set_neuron_tau1(taus_1)


    #set neuron parameters
    #chip.configurator.set_parameter("IF_TAU2_N",3.3e-9)
    #chip.configurator.set_parameter("IF_DC_P",23.9e-11)
    #chip.configurator.set_parameter("VA_EXC_N",2.3e-5)
    #chip.configurator.set_parameter("VDPIE_TAU_P",82.0e-12)
    #chip.configurator.set_parameter("VDPIE_THR_P",82.0e-12)
    #chip.configurator.set_parameter("IF_THR_N",1000.0e-12)

    #chec if the neuron can get excited...
    #index_neu_zero_up = inputpop.synapses['virtual_exc'].addr['neu'] == 244
    #syn = inputpop.synapses['virtual_exc'][index_neu_zero_up]
    #spktrain = syn.spiketrains_regular(100)
    #nsetup.stimulate(spktrain,send_reset_event=False)

    #index_neu_zero_up = inputpop.synapses['programmable'].addr['neu'] == 0
    #syn = inputpop.synapses['programmable'][index_neu_zero_up]
    #spktrain = syn.spiketrains_poisson(600)
    #nsetup.stimulate(spktrain,send_reset_event=False)

else:
    res = L.Reservoir()


###################################################################################
## FUNCTIONS
##################################################################################

def open_wavefile(wavefile):
    spf = wave.open(wavefile,'r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    return signal

def go_reconstruct_signal(duration_rec,upch=300,dnch=305,delta_up=0.1,delta_dn=0.0725, do_plot=True, do_detrend=False):
    '''
    reconstruct signal
    '''         
    out = nsetup.stimulate({},duration=duration_rec)
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
        figure()
        plot(signal_trace_good[:,0],signal_trace_good[:,1])
        
    return signal_trace_good

def go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.0725, do_plot=True, do_detrend=False):
    '''
    go reconstruct signal from output 
    '''                 
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

def det_simple(n_times= 3, duration_rec=1000):
    out_all = []
    for i in range(n_times+1):
        command1 = subprocess.Popen(['sh', '/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/do_stim.sh'])
        out = nsetup.stimulate({},duration=duration_rec,send_reset_event=False)
        if(i>0):
            out_all.append(out)
        command1 = subprocess.Popen(['killall', 'aplay'])
    return out_all

def save_recordings(filename,stim):
    with open(filename, 'wb') as handle:
        pickle.dump(stim, handle)

def load_recordings(filename):
    with open(filename, 'rb') as handle:
        rec = pickle.load(handle)
    return rec

def plot_raster_plot(filename):
    rec = load_recordings(filename)
    n_trial = len(rec)
    for this_trial in range(n_trial):
        rec[this_trial][0].raster_plot()

def plot_determinism(filename,delta_mem=60, sync_bioamp_channel = 305):
    rec = load_recordings(filename)
    n_trial = len(rec)

    fig_h = figure()
    fig_hh = figure()
    # Time vector for analog signals
    Fs    = 100/1e3 # Sampling frequency (in kHz)
    T     = np.max(rec[0][0].raw_data()[:,0])- np.min(rec[0][0].raw_data()[:,0]) #assume all recordings have same lenght 
    nT    = np.round (Fs*T)
    timev = np.linspace(0,T,nT)

    #Conversion from spikes to analog
    membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))

    all_sign = []
    all_out = []
    for this_trial in range(n_trial):
        raw_data = rec[this_trial][0].raw_data()
        raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])

        #use delta as syn
        sync_index = raw_data[:,1] == sync_bioamp_channel
        sync_index = np.where(sync_index)[0]
        time_start = raw_data[sync_index[0],0]
        #delete what happen before the syn index
        index_to_del = np.where(raw_data[:,0] < time_start)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        #delete bioamp spikes
        index_to_del = np.where(raw_data[:,1] > 255)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)
        figure(fig_h.number)
        for i in range(255):
            subplot(16,16,i)
            plot(Y[:,i])
            axis('off')
      
        if (this_trial == 0):
            index_non_zeros = []
            for i in range(256):
                if(np.sum(Y[:,i]) != 0):
                    index_non_zeros.append(i)  
            size_f = np.floor(np.sqrt(len(index_non_zeros)))
        figure(fig_hh.number)
        for i in range(int(size_f**2)):
            subplot(size_f,size_f,i) 
            plot(Y[:,index_non_zeros[i]])  
            axis('off')

def test_determinism(duration_rec, n_trial=3, delta_mem=60, delta_up=15, plot_svd = False, datadir = 'data_fede_format_20020508_pen_3_rec_2', trial=1, sync_bioamp_channel=305):


    data_command = '/home/federico/projects/work/trunk/data/Berkeley/rad-auditory/wehr/Tools/'+datadir+'/do_stim.sh'
    duration_sync = 1600
    
    fig_h = figure()
    fig_hh = figure()
    #figs = figure()
    # Time vector for analog signals
    Fs    = 100/1e3 # Sampling frequency (in kHz)
    T     = duration_rec
    nT    = np.round (Fs*T)
    timev = np.linspace(0,T,nT)

    #Conversion from spikes to analog
    membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
    membrane_up = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_up**2)))

    #nsetup.mapper._program_detail_mapping(2**3+2**4)  
    #command1 = subprocess.Popen(['sh', data_command, str(trial)])
    #out = nsetup.stimulate({},duration=duration_rec, send_reset_event=True)
    #command1 = subprocess.Popen(['killall', 'aplay'])
    all_sign = []
    all_out = []
    for this_trial in range(n_trial):
        command1 = subprocess.Popen(['sh', data_command, str(trial)])
        #time.sleep(1.2)
        out = nsetup.stimulate({},duration=0.2, send_reset_event=True)
        out = nsetup.stimulate({},duration=0.2, send_reset_event=False)
        out = nsetup.stimulate({},duration=duration_rec+duration_sync,send_reset_event=False)
        all_out.append(out)
        signal = go_reconstruct_signal_from_out(out,fig_h,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False, do_plot=False)
        all_sign.append(signal)
        time.sleep(1.2)
        command1 = subprocess.Popen(['killall', 'aplay'])
        #time.sleep(1)
        #nsetup.mapper._program_detail_mapping(0)         
        #use delta as syn
        raw_data = out[0].raw_data()
        sync_index = raw_data[:,1] == sync_bioamp_channel
        sync_index = np.where(sync_index)[0]
        time_start = raw_data[sync_index[0],0]
        #delete what happen before the syn index
        index_to_del = np.where(raw_data[:,0] < time_start+duration_sync)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        #make it start from zero
        raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
        duration = np.max(raw_data[:,0])
        
        index_to_del = np.where(raw_data[:,1] > 255)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)
        figure(fig_h.number)
        for i in range(255):
            subplot(16,16,i)
            plot(Y[:,i])
            axis('off')
      
        if (this_trial == 0):
            index_non_zeros = []
            for i in range(256):
                if(np.sum(Y[:,i]) != 0):
                    index_non_zeros.append(i)  
            size_f = np.floor(np.sqrt(len(index_non_zeros)))
        figure(fig_hh.number)
        for i in range(int(size_f**2)):
            subplot(size_f,size_f,i) 
            plot(Y[:,index_non_zeros[i]])  
            axis('off')

    return all_sign, all_out

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def quote_argument(argument):
    return '"%s"' % (
        argument
        .replace('\\', '\\\\')
        .replace('"', '\\"')
        .replace('$', '\\$')
        .replace('`', '\\`')
    )
    
def poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='g18r2_01-Data', save_data_dir='reservoir_janie', type_stim = 1, save= False):

    
    datadir_tot = '/home/federico/projects/work/trunk/data/Birds/Janie_analog/'+datadir+'/recordings_audio/'
    onlyfiles = [ f for f in listdir(datadir_tot) if isfile(join(datadir_tot,f)) ]
    #extract type stim 
    filename_to_be_used = []
    for i in range(len(onlyfiles)):
        splitted = onlyfiles[i].split('_')
        if( splitted[3] == str(type_stim) ):
            filename_to_be_used.append(onlyfiles[i])    
    
    data_command = '/home/federico/projects/work/trunk/data/Birds/Janie_analog/do_stim.sh'    
    
    duration_sync = 600 
    for this_evoked in range(len(filename_to_be_used)):

        this_evoked_file = datadir_tot + filename_to_be_used[this_evoked]
        command1 = subprocess.Popen(['sh', data_command, str(this_evoked_file)])
        #command1 = subprocess.Popen(['sh', '/home/federico/projects/work/trunk/data/Insects/Insect Neurophys Data/do_teach.sh'])
        out = nsetup.stimulate({},duration=duration_rec+duration_sync, send_reset_event=False)
        command1 = subprocess.Popen(['killall', 'aplay'])
        #signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)

        #use delta as syn
        raw_data = out[0].raw_data()
        sync_index = raw_data[:,1] == sync_bioamp_channel
        sync_index = np.where(sync_index)[0]
        time_start = raw_data[sync_index[2],0]
        #delete what happen before the syn index
        index_to_del = np.where(raw_data[:,0] < time_start)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        #make it start from zero
        #raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
        #duration = np.max(raw_data[:,0])

        #extract input and output
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

        #delete what happen before the syn index
        index_to_del = np.where(raw_data[:,0] < time_start+duration_sync)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        index_to_del = np.where(raw_data_input[:,0] < time_start+duration_sync)
        raw_data_input = np.delete(raw_data_input, index_to_del,axis=0)
        #get again the starting point
        sync_index = raw_data_input[:,1] == 1 
        sync_index = np.where(sync_index)[0]
        time_start = raw_data_input[sync_index[1],0]
        index_to_del = np.where(raw_data_input[:,0] < time_start)
        raw_data_input = np.delete(raw_data_input, index_to_del,axis=0)

        #get again the starting point
        sync_index = raw_data[:,1] == sync_bioamp_channel
        sync_index = np.where(sync_index)[0]
        time_start = raw_data[sync_index[1],0]
        index_to_del = np.where(raw_data[:,0] < time_start)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        effective_start = raw_data[0,0]
        
        #make it start from zero
        raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
        raw_data_input[:,0] = raw_data_input[:,0] - np.min(raw_data_input[:,0])
        
        index_to_del = np.where(raw_data[:,1] > 255)
        raw_data = np.delete(raw_data, index_to_del,axis=0)

        saved_filename = filename_to_be_used[this_evoked].split('.')
        if(save == True):
            np.savetxt(save_data_dir+'/raw_data_input_'+str(datadir)+'_'+saved_filename[0]+'.txt', raw_data_input)
            np.savetxt(save_data_dir+'/raw_data_'+str(datadir)+'_'+saved_filename[0]+'.txt', raw_data)

        out[0].t_start = effective_start 
    return 
    
    
    
def poke_and_save_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='g18r2_01-Data', save_data_dir='reservoir_janie', type_stim = 1):


    
    datadir_tot = '/home/federico/projects/work/trunk/data/Birds/Janie_analog/'+datadir+'/recordings_audio/'
    onlyfiles = [ f for f in listdir(datadir_tot) if isfile(join(datadir_tot,f)) ]
    #extract type stim 
    filename_to_be_used = []
    for i in range(len(onlyfiles)):
        splitted = onlyfiles[i].split('_')
        if( splitted[3] == str(type_stim) ):
            filename_to_be_used.append(onlyfiles[i])    
    
    data_command = '/home/federico/projects/work/trunk/data/Birds/Janie_analog/do_stim.sh'    
    
    duration_sync = 600 
    for this_evoked in range(len(filename_to_be_used)):

        this_evoked_file = datadir_tot + filename_to_be_used[this_evoked]
        command1 = subprocess.Popen(['sh', data_command, str(this_evoked_file)])
        #command1 = subprocess.Popen(['sh', '/home/federico/projects/work/trunk/data/Insects/Insect Neurophys Data/do_teach.sh'])
        out = nsetup.stimulate({},duration=duration_rec+duration_sync, send_reset_event=False)
        command1 = subprocess.Popen(['killall', 'aplay'])
        #signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)

        #use delta as syn
        raw_data = out[0].raw_data()
        sync_index = raw_data[:,1] == sync_bioamp_channel
        sync_index = np.where(sync_index)[0]
        time_start = raw_data[sync_index[2],0]
        #delete what happen before the syn index
        index_to_del = np.where(raw_data[:,0] < time_start)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        #make it start from zero
        #raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
        #duration = np.max(raw_data[:,0])

        #extract input and output
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

        #delete what happen before the syn index
        index_to_del = np.where(raw_data[:,0] < time_start+duration_sync)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        index_to_del = np.where(raw_data_input[:,0] < time_start+duration_sync)
        raw_data_input = np.delete(raw_data_input, index_to_del,axis=0)
        #get again the starting point
        sync_index = raw_data_input[:,1] == 1 
        sync_index = np.where(sync_index)[0]
        time_start = raw_data_input[sync_index[1],0]
        index_to_del = np.where(raw_data_input[:,0] < time_start)
        raw_data_input = np.delete(raw_data_input, index_to_del,axis=0)

        #get again the starting point
        sync_index = raw_data[:,1] == sync_bioamp_channel
        sync_index = np.where(sync_index)[0]
        time_start = raw_data[sync_index[1],0]
        index_to_del = np.where(raw_data[:,0] < time_start)
        raw_data = np.delete(raw_data, index_to_del,axis=0)
        effective_start = raw_data[0,0]
        
        #make it start from zero
        raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
        raw_data_input[:,0] = raw_data_input[:,0] - np.min(raw_data_input[:,0])
        
        index_to_del = np.where(raw_data[:,1] > 255)
        raw_data = np.delete(raw_data, index_to_del,axis=0)

        saved_filename = filename_to_be_used[this_evoked].split('.')
        np.savetxt(save_data_dir+'/raw_data_input_'+str(datadir)+'_'+saved_filename[0]+'.txt', raw_data_input)
        np.savetxt(save_data_dir+'/raw_data_'+str(datadir)+'_'+saved_filename[0]+'.txt', raw_data)

        out[0].t_start = effective_start 
    return 


def poke_and_save(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, trial = 1, datadir='data_fede_format_20020508_pen_3_rec_2', save_data_dir='reservoir_data'):

    duration_sync = 2500 
    #figs = figure()
    data_command = '/home/federico/projects/work/trunk/data/Berkeley/rad-auditory/wehr/Tools/'+datadir+'/do_stim.sh'
    command1 = subprocess.Popen(['sh', data_command, str(trial)])
    #command1 = subprocess.Popen(['sh', '/home/federico/projects/work/trunk/data/Insects/Insect Neurophys Data/do_teach.sh'])
    out = nsetup.stimulate({},duration=duration_rec+duration_sync, send_reset_event=False)
    command1 = subprocess.Popen(['killall', 'aplay'])
    #signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)

    #use delta as syn
    raw_data = out[0].raw_data()
    sync_index = raw_data[:,1] == sync_bioamp_channel
    sync_index = np.where(sync_index)[0]
    time_start = raw_data[sync_index[2],0]
    #delete what happen before the syn index
    index_to_del = np.where(raw_data[:,0] < time_start)
    raw_data = np.delete(raw_data, index_to_del,axis=0)
    #make it start from zero
    #raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
    #duration = np.max(raw_data[:,0])

    #extract input and output
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

    #delete what happen before the syn index
    index_to_del = np.where(raw_data[:,0] < time_start+duration_sync)
    raw_data = np.delete(raw_data, index_to_del,axis=0)
    index_to_del = np.where(raw_data_input[:,0] < time_start+duration_sync)
    raw_data_input = np.delete(raw_data_input, index_to_del,axis=0)
    #get again the starting point
    sync_index = raw_data_input[:,1] == 1 
    sync_index = np.where(sync_index)[0]
    time_start = raw_data_input[sync_index[1],0]
    index_to_del = np.where(raw_data_input[:,0] < time_start)
    raw_data_input = np.delete(raw_data_input, index_to_del,axis=0)

    #get again the starting point
    sync_index = raw_data[:,1] == sync_bioamp_channel
    sync_index = np.where(sync_index)[0]
    time_start = raw_data[sync_index[1],0]
    index_to_del = np.where(raw_data[:,0] < time_start)
    raw_data = np.delete(raw_data, index_to_del,axis=0)
    effective_start = raw_data[0,0]
    
    #make it start from zero
    raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
    raw_data_input[:,0] = raw_data_input[:,0] - np.min(raw_data_input[:,0])
    
    index_to_del = np.where(raw_data[:,1] > 255)
    raw_data = np.delete(raw_data, index_to_del,axis=0)

    np.savetxt(save_data_dir+'/raw_data_input_'+str(datadir)+'_'+str(trial)+'.txt', raw_data_input)
    np.savetxt(save_data_dir+'/raw_data_'+str(datadir)+'_'+str(trial)+'.txt', raw_data)

    out[0].t_start = effective_start 
    return out


def encode_and_poke(duration_rec=15000,delta_mem=25, sync_bioamp_channel=305, smoothing_len=90, trial = 1, what = 'teach', datadir = 'data_fede_format_20020508_pen_3_rec_2',  show_activations = False, save_data = 1, save_data_dir = 'reservoir_data'):
        
    duration_sync = 4000
    #figs = figure()
    data_command = '/home/federico/projects/work/trunk/data/Berkeley/rad-auditory/wehr/Tools/'+datadir+'/do_stim.sh'
    command1 = subprocess.Popen(['sh', data_command, str(trial)])
    #command1 = subprocess.Popen(['sh', '/home/federico/projects/work/trunk/data/Insects/Insect Neurophys Data/do_teach.sh'])
    out = nsetup.stimulate({},duration=duration_rec+duration_sync, send_reset_event=False)
    command1 = subprocess.Popen(['killall', 'aplay'])
    #signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)

    #use delta as syn
    raw_data = out[0].raw_data()
    sync_index = raw_data[:,1] == sync_bioamp_channel
    sync_index = np.where(sync_index)[0]
    time_start = raw_data[sync_index[2],0]
    #delete what happen before the syn index
    index_to_del = np.where(raw_data[:,0] < time_start+duration_sync)
    raw_data = np.delete(raw_data, index_to_del,axis=0)
    #make it start from zero
    raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
    duration = np.max(raw_data[:,0])

    ############# MEMBRANE
    # Time vector for analog signals
    Fs    = 1000/1e3 # Sampling frequency (in kHz)
    T     = duration
    nT    = np.round (Fs*T)
    timev = np.linspace(0,T,nT)
    #Conversion from spikes to analog
    membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
    
    #extract input and output
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
    sync_index = raw_data[:,1] == sync_bioamp_channel
    sync_index = np.where(sync_index)[0]
    time_start = raw_data[sync_index[2],0]
    #delete what happen before the syn index
    index_to_del = np.where(raw_data[:,0] < time_start)
    raw_data = np.delete(raw_data, index_to_del,axis=0)
    #make it start from zero
    raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
    
    index_to_del = np.where(raw_data[:,1] > 255)
    raw_data = np.delete(raw_data, index_to_del,axis=0)

    np.savetxt(save_data_dir+'/raw_data_input_'+str(datadir)+'_'+str(trial)+'.txt', raw_data_input)
    np.savetxt(save_data_dir+'/raw_data_'+str(datadir)+'_'+str(trial)+'.txt', raw_data)

    #teach on reconstructed signal
    X = L.ts2sig(timev, membrane, raw_data_input[:,0], raw_data_input[:,1], n_neu = 256)
    Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)     
    #generate teaching signal orthogonal signal to input
    teach = np.loadtxt('/home/federico/projects/work/trunk/data/Berkeley/rad-auditory/wehr/Tools/'+datadir+'/1_.txt')#
    #teach = np.fromfile(open(trainfile),np.int16)[24:]
    #teach = teach/np.max(teach)
    framepersec = len(teach)/15
    teach = teach[0:framepersec/(duration_rec/1000)]
    #interpolate to match lenght
    signal_ad = [np.linspace(0,duration_rec,len(teach)), teach]
    ynew = np.linspace(0,duration_rec,nT+1)
    s = interpolate.interp1d(signal_ad[0], signal_ad[1],kind="linear")

    teach_sig = s(ynew)     
    teach_sig = np.abs(sigtool.hilbert(teach_sig)) #get envelope
    teach_sig = smooth(teach_sig,window_len=smoothing_len,window='hanning')
 
    if what == 'teach':
        teach_and_plot(X, Y, teach_sig, timev,  show_activations= show_activations)
    if what == 'test':
        test_and_plot(X, Y, teach_sig, timev,  show_activations= show_activations)

def test_and_plot(X, Y, teach_sig, timev, show_activations= False):

    zh = res.predict(Y,Yt=X)

    figure()
    subplot(3,1,1)
    plot(timev,teach_sig[0:len(timev)],label='target signal')
    legend(loc='best')
    subplot(3,1,2)
    plot(timev,zh["input"], label='input')
    plot(timev,teach_sig[0:len(timev)],label='target signal')
    legend(loc='best')
    subplot(3,1,3)
    plot(timev,zh["output"], color='r', label='output')
    plot(timev,teach_sig[0:len(timev)],label='target signal')
    legend(loc='best')
          
    if show_activations == True:    
        figure()        
        index_non_zeros = []
        for i in range(256):
            if(np.sum(Y[:,i]) != 0):
                index_non_zeros.append(i)  
        size_f = np.floor(np.sqrt(len(index_non_zeros)))
        for i in range(int(size_f**2)):
            #subplot(size_f,size_f,i) 
            plot(Y[:,index_non_zeros[i]])  
            #axis('off')      
            
def teach_and_plot(X, Y, teach_sig, timev, show_activations= False):

    res.train(Y,Yt=X,teach_sig=teach_sig[0:len(X),None])#teach_sig[0:len(X),None])
    zh = res.predict(Y,Yt=X)

    figure()
    subplot(3,1,1)
    plot(timev,teach_sig[0:len(timev)],label='teach signal')
    legend(loc='best')
    subplot(3,1,2)
    plot(timev,zh["input"], label='input')
    plot(timev,teach_sig[0:len(timev)],label='teach signal')
    legend(loc='best')
    subplot(3,1,3)
    plot(timev,zh["output"], label='output')
    plot(timev,teach_sig[0:len(timev)],label='teach signal')
    legend(loc='best')
          
    if show_activations == True:    
        figure()        
        index_non_zeros = []
        for i in range(256):
            if(np.sum(Y[:,i]) != 0):
                index_non_zeros.append(i)  
        size_f = np.floor(np.sqrt(len(index_non_zeros)))
        for i in range(int(size_f**2)):
            #subplot(size_f,size_f,i) 
            plot(Y[:,index_non_zeros[i]])  
            #axis('off')  


def encode_and_teach(duration_rec=6000,delta_mem=50, sync_bioamp_channel=305):
        
    #reset previous learning
    res.reset()

    figs = figure()

    command1 = subprocess.Popen(['sh', '/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/do_stim.sh'])
    out = nsetup.stimulate({},duration=duration_rec, send_reset_event=False)
    command1 = subprocess.Popen(['killall', 'aplay'])
    signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)

    #use delta as syn
    raw_data = out[0].raw_data()
    sync_index = raw_data[:,1] == sync_bioamp_channel
    sync_index = np.where(sync_index)[0]
    time_start = raw_data[sync_index[0],0]
    #delete what happen before the syn index
    index_to_del = np.where(raw_data[:,0] < time_start)
    raw_data = np.delete(raw_data, index_to_del,axis=0)
    #make it start from zero
    raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
    duration = np.max(raw_data[:,0])

    ############# MEMBRANE
    # Time vector for analog signals
    Fs    = 1000/1e3 # Sampling frequency (in kHz)
    T     = duration
    nT    = np.round (Fs*T)
    timev = np.linspace(0,T,nT)
    #Conversion from spikes to analog
    membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
    
    #extract input and output
    #raw_data = out[0].raw_data()
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
    sync_index = raw_data[:,1] == sync_bioamp_channel
    sync_index = np.where(sync_index)[0]
    time_start = raw_data[sync_index[0],0]
    #delete what happen before the syn index
    index_to_del = np.where(raw_data[:,0] < time_start)
    raw_data = np.delete(raw_data, index_to_del,axis=0)
    #make it start from zero
    raw_data[:,0] = raw_data[:,0] - np.min(raw_data[:,0])
    
    index_to_del = np.where(raw_data[:,1] > 255)
    raw_data = np.delete(raw_data, index_to_del,axis=0)

    #teach on reconstructed signal
    X = L.ts2sig(timev, membrane, raw_data_input[:,0], raw_data_input[:,1], n_neu = 256)
    Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)     
    #generate teaching signal orthogonal signal to input
    teach = L.orth_signal(X)
    
    res.train(Y,X,teach[0:len(X),None])#teach_sig[0:len(X),None])
    zh = res.predict(Y,X)

    figure()
    subplot(3,1,1)
    plot(timev,teach,label='teach signal')
    legend(loc='best')
    subplot(3,1,2)
    plot(timev,zh["input"], label='input')
    plot(timev,teach,label='teach signal')
    legend(loc='best')
    subplot(3,1,3)
    plot(timev,zh["output"], label='output')
    plot(timev,teach,label='teach signal')
    legend(loc='best')
        
    figure()        
    index_non_zeros = []
    for i in range(256):
        if(np.sum(Y[:,i]) != 0):
            index_non_zeros.append(i)  
    size_f = np.floor(np.sqrt(len(index_non_zeros)))
    for i in range(int(size_f**2)):
        #subplot(size_f,size_f,i) 
        plot(Y[:,index_non_zeros[i]])  
        #axis('off')  

def mini_encoding(duration_rec = 3000, delta_mem = 10):
    #from scipy.fftpack import fft
    #yyf = fft(Y[:,index_non_zeros[i]])

    import wave
    import sys
    spf = wave.open('/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/Hackenelektrodenableitung_mecopoda_elongata_chirper_2_trim.wav','r')
    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()
    time_orig=np.linspace(0, len(signal)/fs, num=len(signal))
    
    
    from scipy import interpolate
    s = interpolate.interp1d(time_orig*1000, signal,kind="linear")
    time_n  = np.linspace(0,np.max(time_orig)*1000,10000)
    ynew = s(time_n)#interpolate.splev(xnew, tck, der=0)
    
    
    #figure()
    #plot(time_orig,signal)
    mul_f = np.ceil(duration_rec/(np.max(time_orig)*1000))
    time_orig = np.linspace(0,np.max(time_n)*mul_f, len(time_n)*mul_f)
    signal_f = []
    for i in range(int(mul_f)):
        signal_f.append(ynew)
    signal = np.reshape(signal_f,[len(ynew)*mul_f])

    figure()
    plot(time_orig,signal)


    #fig_h = figure()
    fig_hh = figure()

    # Time vector for analog signals
    Fs    = 100/1e3 # Sampling frequency (in kHz)
    T     = duration_rec
    nT    = np.round (Fs*T)
    timev = np.linspace(0,T,nT)

    #Conversion from spikes to analog
    membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
 
    out = nsetup.stimulate({},duration=duration_rec)
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
    X = L.ts2sig(timev, membrane, raw_data_input[:,0], raw_data_input[:,1], n_neu = 4)       
    figure()
    for i in range(2):
        plot(X[:,i])
    
    raw_data = out[0].raw_data()
    index_to_del = np.where(raw_data[:,1] > 255)
    raw_data = np.delete(raw_data, index_to_del,axis=0)
    Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)

    index_non_zeros = []
    for i in range(256):
        if(np.sum(Y[:,i]) != 0):
            index_non_zeros.append(i)  
    size_f = np.floor(np.sqrt(len(index_non_zeros)))
    figure(fig_hh.number)
    for i in range(int(size_f**2)):
        #subplot(size_f,size_f,i) 
        plot(Y[:,index_non_zeros[i]])  
        #axis('off')  
    figure()
    for i in range(int(size_f**2)):
        subplot(size_f,size_f,i) 
        plot(Y[:,index_non_zeros[i]])  
        ylim([0,3])
        axis('off')      
        
    figure()
    #raster plot lsm   
    plot(raw_data[:,0], raw_data[:,1], '*', markersize=2)
    ylim([0,256])
    xlim([0,duration_rec])
    xlabel('Time [ms]')
    ylabel('Neu Id')
            
##########################################
## TEST DETERMINISM
#########################################3

#all_sig, all_out = test_determinism(3000, delta_mem=35)

#for i in range(len(all_sig)):
#    aa = all_out[0]
#    aa[0].raster_plot()
#    ylim([0,310])
#    np.savetxt('reservoir/spiketrain_'str(i)'.txt', all_out[i][0].raw_data())

##############################################
## DO EXPERIMENT RESERVOIR
##############################################

### init reservoir
res.reset(alpha=np.logspace(-10,10,500))


print('#################### TEACHING')
#fig_s = figure()
#chip.configurator.set_parameter("NPA_WEIGHT_EXC0_P", 1.5398e-6) #input scaling
#chip.configurator.set_parameter("NPA_WEIGHT_EXC_P", 2.15398e-6) #input scaling
#for this_trial in range(19):
#    encode_and_poke(duration_rec=2000,delta_mem=25, sync_bioamp_channel=305, smoothing_len=100, trial=int(this_trial), what = 'teach', datadir = 'data_fede_format_20020508_pen_3_rec_2', show_activations = False)


def analyze_data():

    root_data = 'reservoir_data/'
    datadir_1 = 'data_fede_format_20020508_pen_3_rec_2'
    datadir_2 = 'data_fede_format_20020608_pen_2_rec_2'
    num_trials = 3
    duration_sync = 4000
    datadir = datadir_1
    what = 'teach'
    sync_bioamp_channel = 305
    delta_mem = 25
    smoothing_len = 100
    duration_rec = 6000
    duration = 6000
    show_activations = True

    for this_trial in range(1,num_trials):
        raw_data_input = np.loadtxt('reservoir_data/data_fede_format_20020508_pen_3_rec_2/raw_data_input_data_fede_format_20020508_pen_3_rec_2_'+str(this_trial)+'.txt')
        raw_data = np.loadtxt('reservoir_data/data_fede_format_20020508_pen_3_rec_2/raw_data_data_fede_format_20020508_pen_3_rec_2_'+str(this_trial)+'.txt')

        ############# MEMBRANE
        # Time vector for analog signals
        Fs    = 1000/1e3 # Sampling frequency (in kHz)
        T     = duration
        nT    = np.round (Fs*T)
        timev = np.linspace(0,T,nT)
        #Conversion from spikes to analog
        membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
        
        #teach on reconstructed signal
        X = L.ts2sig(timev, membrane, raw_data_input[:,0], raw_data_input[:,1], n_neu = 256)
        Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)     
        #generate teaching signal orthogonal signal to input
        teach = np.loadtxt('/home/federico/projects/work/Berkeley_wehr/Tools/'+datadir+'/1_.txt')#
        framepersec = len(teach)/15
        teach = teach[0:framepersec/(duration_rec/1000)]
        #interpolate to match lenght
        signal_ad = [np.linspace(0,duration_rec,len(teach)), teach]
        ynew = np.linspace(0,duration_rec,nT+1)
        s = interpolate.interp1d(signal_ad[0], signal_ad[1],kind="linear")

        teach_sig = s(ynew)     
        teach_sig = np.abs(sigtool.hilbert(teach_sig)) #get envelope
        teach_sig = smooth(teach_sig,window_len=smoothing_len,window='hanning')
     
        if what == 'teach':
            teach_and_plot(X, Y, teach_sig, timev,  show_activations= show_activations)
        if what == 'test':
            test_and_plot(X, Y, teach_sig, timev,  show_activations= show_activations)




#poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='b20r1_01-Data', save_data_dir='reservoir_janie/b20r1_01-Data/recordings_chip/', type_stim = 3, save= True)

#poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='b20r1_02-Data', save_data_dir='reservoir_janie/b20r1_02-Data/recordings_chip/', type_stim = 2, save= True)
#poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='b20r1_02-Data', save_data_dir='reservoir_janie/b20r1_02-Data/recordings_chip/', type_stim = 3, save= True)
#poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='b20r1_02-Data', save_data_dir='reservoir_janie/b20r1_02-Data/recordings_chip/', type_stim = 4, save= True)

##poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='g4r4_01-Data', save_data_dir='reservoir_janie/g4r4_01-Data/recordings_chip/', type_stim = 2, save= True)
#poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='g4r4_01-Data', save_data_dir='reservoir_janie/g4r4_01-Data/recordings_chip/', type_stim = 3, save= True)
##poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='g4r4_01-Data', save_data_dir='reservoir_janie/g4r4_01-Data/recordings_chip/', type_stim = 4, save= True)

##poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='g18r2_02-Data', save_data_dir='reservoir_janie/g18r2_02-Data/recordings_chip/', type_stim = 2, save= True)
#poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='g18r2_02-Data', save_data_dir='reservoir_janie/g18r2_02-Data/recordings_chip/', type_stim = 3, save= True)
##poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='g18r2_02-Data', save_data_dir='reservoir_janie/g18r2_02-Data/recordings_chip/', type_stim = 4, save= True)
