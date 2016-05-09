#!/usr/env/python
#
# author federico corradi
# federico@ini.phys.ethz.ch
#
import numpy as np
from pylab import *
import sys
import time
import scipy.signal
import subprocess
import scipy.signal.signaltools as sigtool
import wave
from os import listdir
from os.path import isfile, join
from scipy.io import wavfile
import random
#sys.path.append('/home/federico/project/work/api/perceptrons_liquid/')
sys.path.append('/home/federico/space/datafiles/code_and_data/python/mn256r1_ncs/api/wij/')
#import pyAgilent
import matplotlib
from pylab import *
import cPickle

#from perceptrons_liquid import Perceptrons
from wij import SynapsesLearning

sys.path.append('/home/federico/space/datafiles/code_and_data/python/mn256r1_ncs/api/reservoir/')
sys.path.append('/home/federico/space/datafiles/code_and_data/python/mn256r1_ncs/api/bioamp/')

from scipy import interpolate
import reservoir as L
label_size = 14
rcParams['xtick.labelsize'] = label_size 
rcParams['ytick.labelsize'] = label_size 

ioff()

use_chip = False 
load_config = False
config_dir = 'data_reservoir/ee03-ii02_b4/'
wavefile =  'Hackenelektrodenableitung_mecopoda_elongata_chirper_2.wav'

num_channels = 64 
duration = 2
do_learning_exp = False
n_learning_trials = 1 
clock_neu = 255

####
# LOAD RESULTS
####
load_results = False
load_directory = '~/space/datafiles/code_and_data/python/spkInt/mn256r1/experiments/bioamp_delta/learning_trials/2015-06-11_00-34-22/'#'learning_trials/2015-06-11_02-27-30/'
make_pic_tbcas = False
plot_hist_tot = True
dirr = 'learning_trials_got/'
dirra = 'data_OK/'
load_after_analysis = False
plot_testing = False
plot_inputs = False

features_neu = np.linspace(0,(num_channels*2)-1,num_channels*2)
perceptron_neu = np.linspace(128,255,128)

if use_chip:
    from bioamp import Bioamp
    import pyNCS

    ##########################################
    # INIT PYNCS and CHIP
    ########################################

    prefix= '/home/federico/projects/work/trunk/code/python/mn256r1_ncs/'  
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
    #if(load_config == True):
    #    res.load_config(config_dir)
    #res.program_config()    
    res.reset(alpha=np.logspace(-10,100,500))  ##init regressor

    #######################################################3
    ####  PERCEPTRONS
    ########################################################
    
    #make on chip network
    feature_pop = pyNCS.Population('', '')
    perceptron_pop = pyNCS.Population('', '')
    feature_pop.populate_by_id(nsetup, 'mn256r1', 'excitatory', features_neu)
    perceptron_pop.populate_by_id(nsetup, 'mn256r1', 'excitatory', perceptron_neu)

    net = Perceptrons(perceptron_pop,feature_pop) 
    net.matrix_learning_pot[:] = 0
    net.upload_config()
    #test
    #syn = feature_pop.synapses['programmable'][::16]
    #stim = syn.spiketrains_poisson(10)
    #nsetup.stimulate(stim,send_reset_event=False)

    #set up filters and connect retina
    inputpop = pyNCS.Population('','')
    inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  

    sl = SynapsesLearning(inputpop, 'learning')

    ################################
    ### CONNECT BIOAMP TO RESERVOIR
    ################################

    #bioamp._init_fpga_mapper()
    #bioamp.map_bioamp_reservoir_broadcast(n_columns=4)
    #nsetup.mapper._program_detail_mapping(2**3+2**4)  


    #print('load parameters from bioamp interface.. use file biases/bioamp_delta_reservoir.txt or biases/bioamp_delta_reservoir_forest.txt') 
    #raw_input("Press Enter to when done...")

    #set programmable synapse parameters
    #chip.configurator.set_parameter("NPA_WEIGHT_INH0_N",0.001e-6)
    #chip.configurator.set_parameter("NPA_WEIGHT_INH1_N",1.2625e-6)
    #chip.configurator.set_parameter("NPA_WEIGHT_INH_N",0.03025e-6)
    #chip.configurator.set_parameter("NPA_WEIGHT_EXC_P",1.610925e-6)
    #chip.configurator.set_parameter("NPA_WEIGHT_EXC0_P",1.5398e-6)
    #chip.configurator.set_parameter("NPA_WEIGHT_EXC1_P",0.5e-6)
    #chip.configurator.set_parameter("NPDPIE_THR_P",10e-12)
    #chip.configurator.set_parameter("NPDPIE_TAU_P",80e-12)
    #chip.configurator.set_parameter("NPDPII_THR_P",200e-12)
    #chip.configurator.set_parameter("NPDPII_TAU_P",200e-12)

    #neuron parameters
    #chip.configurator.set_parameter("IF_TAU2_N",8.6e-12)
    #chip.configurator.set_parameter("IF_TAU1_N",0.033e-6)

    # clock on
    chip.configurator._set_all_neu_tau2()
    taus_1 = np.array([clock_neu])
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

res.reset(alpha=np.logspace(-10,10,300))

#rhoW = max(abs(linalg.eig(res.matrix_programmable_w)[0]))
#scaled_matrix_programmable_w *= 1.25 / rhoW   #no meaning

#res.reset(alpha=np.logspace(-8,8,50))
#res.reset(alpha=np.logspace(-10,10,500)) #init with right alphas

###############
# FUNCTIONS
###############
def ismember( a,b):
	'''
	as matlab: ismember
	'''
	# tf = np.in1d(a,b) # for newer versions of numpy
	tf = np.array([i in b for i in a])
	u = np.unique(a[tf])
	index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
	return tf, index

def get_scope():
    osc_a = pyAgilent.Agilent(host="172.19.10.159");
    osc_a._send_command('WAV:FORM RAW');
    #osc_a.single()
    membrane = osc_a._read_data_from_channel(1)
    weight = osc_a._read_data_from_channel(4)
    time_w = np.linspace(0,10000,len(weight))
    time_membrane = np.linspace(0,10000,len(membrane))
    figure()
    subplot(2,1,1)
    plot(time_membrane, membrane, '-', label='Neu: 189')
    #xlim([2500,8500])
    legend(loc='best')
    #xlabel('Time [ms]', fontsize=18)
    ylabel('Vmem [V]', fontsize=18)
    subplot(2,1,2)
    plot(time_w, weight, '-', label='Synapse: pre-id 13 post-id 189')
    #xlim([2500,8500])
    ylim([0,1.8])
    legend(loc='best')
    xlabel('Time [ms]', fontsize=18)
    ylabel('Vw [V]', fontsize=18)

def go_reconstruct_signal(raw_input_multi_channel_3_l,upch=0,dnch=1,delta_up=0.1,delta_dn=0.1,do_detrend=False,do_plot=False):
    '''
    reconstruct signal
    '''         
    raw_data_input = raw_input_multi_channel_3_l

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

def get_recordings_files(reservoirdir, datadir, what='teach', num_channels=32, trial=[1,2,3],type_stim=4,  mix_prob=[0.33,0.34,0.33]):

    directory = reservoirdir+'/'+datadir+'/recordings_chip'
    onlyfiles = [ f for f in listdir(directory) if isfile(join(directory,f)) ]
    onlyfiles.sort()
    num_rec_per_trial = np.round(np.dot(num_channels,mix_prob)) 
    #onlyfiles = np.array(onlyfiles)
    #np.random.shuffle(onlyfiles)
    onlyfiles_fin = []
    ok_got = 0
    all_trials = trial
    braker = False
    counter_dim_a = 1
    counter_dim_b = 1
    for trial in range(len(all_trials)):         
        for i in range(len(onlyfiles)):
            if(len(onlyfiles_fin) == num_channels):
                breaker= True
                break
            if(ok_got == num_rec_per_trial[trial]):
                if(trial < len(all_trials)):
                    trial = trial+1;
                    ok_got = 0
            if(trial == len(all_trials)):
                breaker = True 
                break
            this_file = onlyfiles[i].split('_')
            if( this_file[2] != 'input' ):
                if( this_file[7] == str(type_stim)):
                    if( this_file[5] == str(all_trials[trial])):
                        print(onlyfiles[i])
                        onlyfiles_fin.append(onlyfiles[i])
                        ok_got += 1

 
        if braker:
            break
    onlyfiles_fin = np.array(onlyfiles_fin)
    #np.random.shuffle(onlyfiles_fin)
    return onlyfiles_fin

def get_recordings_files_old(reservoirdir, datadir, what='teach', num_channels=32, type_stim=4, trial = [1]):
    directory = reservoirdir+'/'+datadir+'/recordings_chip'
    onlyfiles = [ f for f in listdir(directory) if isfile(join(directory,f)) ]
    onlyfiles.sort()

    onlyfiles_train = []
    onlyfiles_test = []
    ok_got = 0
    all_trials = trial
    for trial in range(len(all_trials)):         
        for i in range(len(onlyfiles)):
            if(ok_got == num_channels*2):
                break
            this_file = onlyfiles[i].split('_')
            if( this_file[2] != 'input' ):
                if( this_file[7] == str(type_stim)):
                    if( this_file[5] == str(all_trials[trial])):
                        if(ok_got < num_channels):
                            print(onlyfiles[i])
                            onlyfiles_train.append(onlyfiles[i])
                        else:
                            onlyfiles_test.append(onlyfiles[i])
                        ok_got += 1
    
    
    if(what != 'teach'):
        onlyfiles = onlyfiles_train
    else:
        onlyfiles = onlyfiles_test
    return onlyfiles

def plot_input_files(files_3_l, reservoirdir='reservoir_janie', datadir='g18r2_01-Data', color='blue'):

    n_chan = np.shape(files_3_l)[0]
         
    figure()        
    title('datadir '+datadir)
    for i in range(n_chan):        
        ts = files_3_l[i].split('raw_data_'+datadir+'_')
        if(len(ts) > 1):
            ff = ts[1]
        else:
            ff = ts[0]
        raw_trace = np.loadtxt(reservoirdir+'/'+datadir+'/recordings/'+ff)
        fs = int(files_3_l[i].split('_')[13])
        duration_s = len(raw_trace)/float(fs)
        time_s = np.linspace(0,duration_s,len(raw_trace))
        subplot(n_chan,1,i)
        plot(time_s,raw_trace,color=color)
        #axvspan(zoomed[0]/1000.0, zoomed[1]/1000.0, color='red', alpha=0.5)
        xlim([0,4])
        axis('off')


def plot_raw_input_multichannel(raw_input_multi_channel_3_l, files_3_l, n_chan=None, reservoirdir='reservoir_janie', datadir='g18r2_01-Data', zoomed=[500,750], plot_delta=False):


    if(n_chan == None):
        n_chan = np.shape(raw_input_multi_channel_3_l)[0]
        
    figure(); 
    for i in range(n_chan):
            trace = go_reconstruct_signal(raw_input_multi_channel_3_l[i],upch=0,dnch=1,delta_up=0.1,delta_dn=0.1, do_plot=False, do_detrend=True)
            subplot(n_chan,1,i)
            plot(trace[:,0],trace[:,1])
            axvspan(zoomed[0], zoomed[1], color='red', alpha=0.5)
            ylim([np.min(trace[:,1]), np.max(trace[:,1])])
            xlim([0,4000])
            axis('off')
            
    if(plot_delta):        
        figure()
        ci = n_chan
        for i in range(n_chan):
            index_up = raw_input_multi_channel_3_l[i][:,1] == 1
            index_dn = raw_input_multi_channel_3_l[i][:,1] == 0
            vlines(raw_input_multi_channel_3_l[i][index_up,0],raw_input_multi_channel_3_l[i][index_up,1]+(ci), raw_input_multi_channel_3_l[i][index_up,1]+(ci+0.25), color='b')     
            vlines(raw_input_multi_channel_3_l[i][index_dn,0],raw_input_multi_channel_3_l[i][index_dn,1]+(ci+0.5), raw_input_multi_channel_3_l[i][index_dn,1]+(ci+0.5+0.25), color='r')    
            xlim([zoomed[0],zoomed[1]])
            xlabel('Time [ms]', fontsize=18)
            ylabel('Channel num', fontsize=18)
            ci = ci - 1
    
    figure()        
    for i in range(n_chan):        
        ts = files_3_l[i].split('raw_data_'+datadir+'_')
        raw_trace = np.loadtxt(reservoirdir+'/'+datadir+'/recordings/'+ts[1])
        fs = int(files_3_l[i].split('_')[13])
        duration_s = len(raw_trace)/float(fs)
        time_s = np.linspace(0,duration_s,len(raw_trace))
        subplot(n_chan,1,i)
        plot(time_s,raw_trace)
        #axvspan(zoomed[0]/1000.0, zoomed[1]/1000.0, color='red', alpha=0.5)
        xlim([0,4])
        axis('off')
        
def get_all_recordings(reservoirdir,datadir,what='teach',num_channels=32, type_stim=4, trial = 1):
    directory = reservoirdir+'/'+datadir+'/recordings_chip';
    onlyfiles = [ f for f in listdir(directory) if isfile(join(directory,f)) ];
    onlyfiles.sort()

    onlyfiles_fin = []
    ok_got = 0
    for i in range(len(onlyfiles)):
        if(ok_got == num_channels*2):
            break
        this_file = onlyfiles[i].split('_')
        if( this_file[2] != 'input' ):
            if( this_file[7] == str(type_stim)):
                if( this_file[5] == str(trial)):
                    onlyfiles_fin.append(onlyfiles[i])
                    ok_got += 1
    
    onlyfiles = onlyfiles_fin
    return onlyfiles


def get_all_stimuli(reservoirdir,datadir, what='teach',num_channels=32, type_stim=4, trial= 1):
    directory = reservoirdir+'/'+datadir+'/recordings_chip';
    onlyfiles = [ f for f in listdir(directory) if isfile(join(directory,f)) ];
    onlyfiles.sort()

    onlyfiles_fin = []
    ok_got = 0
    for i in range(len(onlyfiles)):
        if(ok_got == num_channels*2):
            break
        this_file = onlyfiles[i].split('_')
        if( this_file[2] != 'input' ):
            if( this_file[7] == str(type_stim)):
                if( this_file[5] == str(trial)):
                    stimulus_file = onlyfiles[i].split('raw_data_'+datadir)
                    stimulus_file = 'stim'+stimulus_file[1]
                    onlyfiles_fin.append(stimulus_file)
                    ok_got += 1
    
    onlyfiles = onlyfiles_fin
    return onlyfiles
    
def plot_spectrogram_stim(type_stim=4, reservoirdir='reservoir_janie', datadir='g18r2_01-Data', trial=1,  num_channels=32):
    '''
    plot input sound spectrogram
    '''
    onlyfiles = get_all_stimuli(reservoirdir,datadir, num_channels=num_channels, type_stim=type_stim)
    #random.shuffle(onlyfiles)
    ok_got = 0
    for i in range(len(onlyfiles)):
        this_file = onlyfiles[i].split('_')
        if( this_file[4] == str(type_stim) ):
            #print(this_file[4])
            ok_got +=1
            filename = reservoirdir+'/'+datadir+'/stimuli/'+onlyfiles[i]
            fs, audio = scipy.io.wavfile.read(filename)
            figure()
            specgram(audio, Fs= fs, scale_by_freq=False, sides='default')  
            xlabel('Time [s]', fontsize=18)    
            ylabel('Frequency [Hz]', fontsize=18)  
            xlim([0,4])  
            colorbar()
            if(ok_got == trial):
                break      
                
    return 

def mean_neu_firing(spike_train, n_neurons,nbins=10):
    simulation_time = [np.min(spike_train[:,0]),np.max(spike_train[:,0])]
    un, bins = np.histogram(simulation_time,nbins)
    mean_rate = np.zeros([len(n_neurons),nbins])
    for b in range(nbins):
        #simulation_time = [np.min(spike_train[0][:]), np.max(spike_train[0][:])]
        for i in range(len(n_neurons)):
            index_neu = np.where(np.logical_and(spike_train[:,1] == n_neurons[i], np.logical_and(spike_train[:,0] >     bins[b] , spike_train[:,0] < bins[b+1] )) )
            mean_rate[i,b] = (len(index_neu[0])*1000.)/(bins[b+1]-bins[b]) # time unit: ms
    return mean_rate

def find_class_teached(out_teacher_class_b):
    '''
       teached class
    '''
    out = out_teacher_class_b
    raw_data = out[0].raw_data()
    index_perc, un = ismember(raw_data[:,1], perceptron_neu)
    mean_rates = mean_neu_firing(raw_data[index_perc],perceptron_neu,nbins=1)
    mean_rates = np.reshape(mean_rates,(-1))
    argss = mean_rates.argsort()
    class_a = argss[0:len(argss)/2]
    class_b = argss[len(argss)/2::]

    return perceptron_neu[class_a], perceptron_neu[class_b]


def get_out_data_hist(out_4_arr, class_a_perc, class_b_perc, num=100):
    '''
    one pop hisogram
    '''
    electrode_ids = features_neu
    raw_data = out_4_arr[0][0].raw_data()
    t_start = np.min(raw_data[:,0])
    raw_data[:,0] = raw_data[:,0] - t_start
    index_class_a, un = ismember(raw_data[:,1], class_a_perc)
    index_class_b, un = ismember(raw_data[:,1], class_b_perc)
    index_reservoir, un = ismember(raw_data[:,1],   electrode_ids)
    reservoir_index_up = np.unique(raw_data[index_reservoir,1])
    class_a = np.unique(raw_data[index_class_a,1])
    class_b = np.unique(raw_data[index_class_b,1])
    mean_tot_class_a = []
    mean_tot_class_b = []
    for i in range(len(out_4_arr)):
        mean_rate_class_a = mean_neu_firing(raw_data[index_class_a], class_a_perc,nbins=1)
        mean_rate_class_b = mean_neu_firing(raw_data[index_class_b], class_b_perc,nbins=1)
        mean_tot_class_b.append(np.mean(mean_rate_class_b))
        mean_tot_class_a.append(np.mean(mean_rate_class_a))

    #figure(num=num)
    #hist(mean_tot_class_b, color='red')
    #hist(mean_tot_class_a, color='green')

	return mean_tot_class_b, mean_tot_class_a

def plot_out(out_4, class_a_perc, class_b_perc):
    '''
    make paper plots
    '''
    electrode_ids = features_neu
    raw_data = out_4[0].raw_data()
    t_start = np.min(raw_data[:,0])
    raw_data[:,0] = raw_data[:,0] - t_start
    index_class_a, un = ismember(raw_data[:,1], class_a_perc)
    index_class_b, un = ismember(raw_data[:,1], class_b_perc)
    index_reservoir, un = ismember(raw_data[:,1],   electrode_ids)
    reservoir_index_up = np.unique(raw_data[index_reservoir,1])
    class_a = np.unique(raw_data[index_class_a,1])
    class_b = np.unique(raw_data[index_class_b,1])
    fig = figure(figsize=(8, 6)) 
    gs = GridSpec(3, 2, width_ratios=[4, 1]) 
    subplot(gs[0])
    plot(raw_data[index_reservoir,0], raw_data[index_reservoir,1], 'go', markersize=1.2)
    xlim([0, 4000])
    ylim([0,len(features_neu)])
    ylabel('Reservoir', fontsize=18)
    subplot(gs[1])
    aa = hist(raw_data[index_reservoir,1],len(features_neu), orientation='horizontal')
    ylim([0,len(features_neu)])
    xlim([0,np.max(aa[0])+200])
    xticks([0,np.max(aa[0])+200])
    yticks([0,len(features_neu)])
    subplot(gs[2])
    hist_data = []
    for i in range(len(class_a)):
        this_index = np.where(raw_data[index_class_a,1] == class_a[i])[0]
        hist_data.extend( np.repeat(i,len(index_class_a[this_index])))
        plot(raw_data[index_class_a,0][this_index], np.repeat(i,len(index_class_a[this_index])), 'yo', markersize=1.2)
    ylabel('Perc. class A', fontsize=18)
    ylim([0,len(class_a)+1])
    xlim([0, 4000])
    subplot(gs[3])
    if(len(hist_data) > 1):
        hist(hist_data,len(class_a)+1, orientation='horizontal')
        aab = histogram(hist_data, 64)
        vlines(np.mean(aab[0]), 0,len(features_neu), color='red', linewidth=2)
        text(np.mean(aab[0])+10,len(class_a)/2, 'mean rate: '+str(np.mean(aab[0])/4.0) , fontsize=18)
        ylim([0,len(class_a)+1])
        xlim([0, np.max(aa[0])+200])
        xticks([0,np.max(aa[0])+200])
        yticks([0,len(class_a)])
    subplot(gs[4])
    hist_data = []
    for i in range(len(class_b)):
        this_index = np.where(raw_data[index_class_b,1] == class_b[i])[0]
        hist_data.extend(np.repeat(i,len(index_class_b[this_index])))
        plot(raw_data[index_class_b,0][this_index], np.repeat(i,len(index_class_b[this_index])), 'ro', markersize=1.2)
        ylabel('Perc. class B', fontsize=18)
        ylim([0,len(class_b)+1])
        xlim([0, 4000])
        xlabel('Time [ms]', fontsize=18)
    subplot(gs[5])
    if(len(hist_data) > 1):
        hist(hist_data,len(class_b)+1, orientation='horizontal')
        aab = histogram(hist_data, 64)
        vlines(np.mean(aab[0]), 0,len(features_neu), color='red', linewidth=2)
        text(np.mean(aab[0])+10,len(class_b)/2, 'mean rate: '+str(np.mean(aab[0])/4.0) , fontsize=18)
        xlabel('Num. spikes',  fontsize=18)
        xlim([0, np.max(aa[0])+200])
        xticks([0,np.max(aa[0])+200])
        yticks([0,len(class_b)])
        ylim([0,len(class_b)+1])


def test_and_plot(X, Y, teach_sig, timev, show_activations= False , teacher = None):

    zh = res.predict(Y,Yt=X)

    figure()
    subplot(3,1,1)
    plot(timev,teacher[0:len(timev)],label='target signal')
    legend(loc='best')
    subplot(3,1,2)
    plot(timev,zh["input"], label='input')
    plot(timev,teacher[0:len(timev)],label='target signal')
    legend(loc='best')
    subplot(3,1,3)
    plot(timev,zh["output"], color='r', label='output')
    plot(timev,teacher[0:len(timev)],label='target signal')
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
            
def teach_and_plot(X, Y, teach_sig, timev, show_activations= False, teacher = None):

    res.train(Y,Yt=X,teach_sig=teacher[0:len(X),None])#teach_sig[0:len(X),None])
    zh = res.predict(Y,Yt=X)

    figure()
    subplot(3,1,1)
    plot(timev,teacher[0:len(timev)],label='teach signal')
    legend(loc='best')
    subplot(3,1,2)
    plot(timev,zh["input"], label='input')
    plot(timev,teacher[0:len(timev)],label='teach signal')
    legend(loc='best')
    subplot(3,1,3)
    plot(timev,zh["output"], label='output')
    plot(timev,teacher[0:len(timev)],label='teach signal')
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

def read_syn_matrix_learning():
              
    matrix_rec = np.zeros([256,256])
    chip.configurator.set_parameter("IF_TAU2_N", 8.6e-7)
    nsetup.mapper._program_onchip_plastic_connections(matrix_rec)
    nsetup.mapper._program_onchip_programmable_connections(matrix_rec)
    time.sleep(0.2)
    sl.get_br()
    time.sleep(0.2)

def test_perceptrons(stim, perceptron_hi = np.linspace(127,127+31,32), teacher_freq=250, noise_freq=75, duration=2000):

    chip.configurator.set_parameter("IF_TAU2_N", 8.6e-7)
    chip.configurator.set_parameter("PDPI_TAU_P", 2.4e-05)
    chip.configurator.set_parameter("PDPI_THR_P", 0.0)
    chip.configurator.set_parameter("VDPIE_THR_P", 0.0)
    chip.configurator.set_parameter("VDPIE_TAU_P", 2.0e-05)
    chip.configurator.set_parameter("VDPII_TAU_N", 2.0e-05)
    chip.configurator.set_parameter("VDPII_THR_N", 0.0)      
    time.sleep(0.5)
    nsetup.chips['mn256r1'].load_parameters('biases/biases_wijtesting_bioamp_perceptrons.biases')
    time.sleep(0.5)

    ##train 
    stimulus = {1:stim}
    
    #teacher hi
    tf, un = net._ismember(inputpop.synapses['virtual_exc'].addr['neu'], perceptron_hi)
    syn_teacher = inputpop.synapses['virtual_exc'][tf]
    teacher = syn_teacher.spiketrains_regular(30,duration= 3000)
    
    #teacher low
    tf, un = net._ismember(perceptron_neu, perceptron_hi)
    tf_noise = np.logical_not(tf)
    tf, un = net._ismember(inputpop.synapses['virtual_exc'].addr['neu'], perceptron_neu[tf_noise])
    syn_noise = inputpop.synapses['virtual_exc'][tf]
    noise = syn_noise.spiketrains_regular(30,duration= 3000)
    
    stim_final = pyNCS.pyST.merge_sequencers({1:stim},teacher)
    stim_final = pyNCS.pyST.merge_sequencers(stim_final, noise)
    
    #out = nsetup.stimulate({},send_reset_event=True)
    #out = nsetup.stimulate({},send_reset_event=False) #should be nice to know why...
    out = nsetup.stimulate(stim_final, send_reset_event=False)   

    chip.configurator.set_parameter("PA_DELTADN_N", 0.0)
    chip.configurator.set_parameter("PA_DELTAUP_P", 0.0)
    
    return out

def train_perceptrons(stim, perceptron_hi = np.linspace(127,127+31,32), teacher_freq=250, noise_freq=75, duration=2000):

    chip.configurator.set_parameter("IF_TAU2_N", 8.6e-7)
    chip.configurator.set_parameter("PDPI_TAU_P", 2.4e-05)
    chip.configurator.set_parameter("PDPI_THR_P", 0.0)
    chip.configurator.set_parameter("VDPIE_THR_P", 0.0)
    chip.configurator.set_parameter("VDPIE_TAU_P", 2.0e-05)
    chip.configurator.set_parameter("VDPII_TAU_N", 2.0e-05)
    chip.configurator.set_parameter("VDPII_THR_N", 0.0)      
    time.sleep(0.5)
    nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_bioamp_perceptrons.biases')
    time.sleep(0.5)

    ##train 
    stimulus = {1:stim}
    
    #teacher hi
    tf, un = net._ismember(inputpop.synapses['virtual_exc'].addr['neu'], perceptron_hi)
    syn_teacher = inputpop.synapses['virtual_exc'][tf]
    teacher = syn_teacher.spiketrains_poisson(teacher_freq,duration= 3000)
    
    #teacher low
    tf, un = net._ismember(perceptron_neu, perceptron_hi)
    tf_noise = np.logical_not(tf)
    tf, un = net._ismember(inputpop.synapses['virtual_exc'].addr['neu'], perceptron_neu[tf_noise])
    syn_noise = inputpop.synapses['virtual_exc'][tf]
    noise = syn_noise.spiketrains_poisson(noise_freq,duration= 3000)
    
    stim_final = pyNCS.pyST.merge_sequencers({1:stim},teacher)
    stim_final = pyNCS.pyST.merge_sequencers(stim_final, noise)
    
    #out = nsetup.stimulate({},send_reset_event=True)
    #out = nsetup.stimulate({},send_reset_event=False) #should be nice to know why...
    out = nsetup.stimulate(stim_final, send_reset_event=False)   

    chip.configurator.set_parameter("PA_DELTADN_N", 0.0)
    chip.configurator.set_parameter("PA_DELTAUP_P", 0.0)
    
    return out

def score_class(out_4, class_a_perc, class_b_perc, threshold_error=100):

    raw_data = out_4[0].raw_data()
    t_start = np.min(raw_data[:,0])
    out = out_4[0]
    out.t_start = t_start
    mean_rates = out.firing_rate(time_bin=150)
    class_a_perc = np.array(class_a_perc, dtype=int)
    class_b_perc = np.array(class_b_perc, dtype=int)
    class_a_mean = mean_rates[class_a_perc]
    class_b_mean = mean_rates[class_b_perc]
    a,b = np.where(class_b_mean > threshold_error)
    class_b_mean[a] = 1 
    a,b = np.where(class_a_mean > threshold_error)
    class_a_mean[a] = 1
    avr_a = np.sum(np.sum(class_a_mean,axis=1))
    print('class down: ', avr_a)
    avr_b = np.sum(np.sum(class_b_mean,axis=1))
    print('class up: ', avr_b)

    return avr_a, avr_b

def do_testing_perceptron_exp(n_channels=16, type_stim_a = 4, syntype='virtual',  type_stim_b=3, teacher_freq=180, noise_freq=3, class_a_perc = np.linspace(127,127+63,64), class_b_perc = np.linspace(127+64,127+64+63,64), n_syn=2, trial = [1], datadir='g18r2_01-Data', mix_prob=[1]): 

    out_4, raw_input_multi_channel_4, stim_4, files_4 = record_fake_multi_electrode_array(what='test', n_channels = n_channels, type_stim=type_stim_a,  perceptron_hi = class_a_perc,teacher_freq=teacher_freq,  noise_freq=noise_freq , syntype=syntype, n_syn=n_syn, trial = trial, datadir=datadir, mix_prob=mix_prob)
    avr_a, avr_b = score_class(out_4, class_a_perc, class_b_perc)
    if(avr_a > avr_b):
        print("#############  A")
    else:
        print("#############  B")

    out_3, raw_input_multi_channel_3, stim_3, files_3 = record_fake_multi_electrode_array(what='test', n_channels = n_channels, type_stim=type_stim_b,  perceptron_hi = class_b_perc, teacher_freq=teacher_freq,  noise_freq=noise_freq, syntype=syntype, n_syn=n_syn, datadir=datadir, mix_prob=mix_prob, trial=trial)
    avr_a, avr_b = score_class(out_3, class_a_perc, class_b_perc)
    if(avr_a > avr_b):
        print("#############  A")
    else:
        print("#############  B")

    return out_4, out_3, raw_input_multi_channel_4, raw_input_multi_channel_3, stim_4, stim_3, files_4, files_3

def create_stimuli_train(n_channels=32, type_stim=4, trial = [2,3,4,5], mix_prob=[0.33,0.33,0.15,0.15], nstim=1, datadir= 'g18r2_01-Data'):
    '''
        create stimuli teach and test what is not used anymore
    '''
    files_tot = []
    all_raw = []
    for i in range(nstim):
        raw_input_multi_channel, files = make_multi_electrode_input(what='teach',num_channels=n_channels, type_stim=type_stim, trial = trial, mix_prob = mix_prob, datadir=datadir)
        all_raw.append(raw_input_multi_channel)
        files_tot.append(files)
        
    return all_raw, files_tot

def create_spike_train_from_files(files, reservoirdir='reservoir_janie', datadir='g18r2_01-Data'):
    all_raw_multi_channels = []
    all_spike_trains_multi_channels = []
    n_trials = len(files)
    for i in range(n_trials):
        raw_input_multi_channel = make_multi_electrode_input_from_files(files,reservoirdir,datadir)
        spike_train = create_spike_train_from_raw(raw_input_multi_channel)
        all_raw_multi_channels.append(raw_input_multi_channel)
        all_spike_trains_multi_channels.append(spike_train)
        
    return all_spike_trains_multi_channels, all_raw_multi_channels       
    

def do_learning_perceptron_exp(n_channels=16, type_stim_a = 4, syntype='virtual',  type_stim_b=3, teacher_freq=180, noise_freq=3, class_a_perc = np.linspace(127,127+63,64), class_b_perc = np.linspace(127+64,127+64+63,64), n_syn=2, trial = [1,2,3], mix_prob = [0.33,0.33,0.33],  datadir= 'g18r2_01-Data'):
    
    out_4, raw_input_multi_channel_4, stim_4, files_4  = record_fake_multi_electrode_array(what='teach', n_channels = n_channels, type_stim=type_stim_a,  perceptron_hi = class_a_perc,teacher_freq=teacher_freq,  noise_freq=noise_freq , syntype=syntype, n_syn=n_syn,  trial = trial, mix_prob=mix_prob, datadir=datadir)
    out_3, raw_input_multi_channel_3, stim_3, files_3 = record_fake_multi_electrode_array(what='teach', n_channels = n_channels, type_stim=type_stim_b,  perceptron_hi = class_b_perc, teacher_freq=teacher_freq,  noise_freq=noise_freq, syntype=syntype, n_syn=n_syn, trial = trial, mix_prob=mix_prob, datadir=datadir)

    return out_4, out_3, raw_input_multi_channel_4, raw_input_multi_channel_3, stim_4, stim_3, files_4, files_3
    
def record_fake_multi_electrode_array(what='teach', n_channels = 8, type_stim=4, n_syn = 4, use_broadcast = False, datadir='g18r2_01-Data', save_data_dir='reservoir_janie', syntype='virtual', perceptron_hi = np.linspace(127,127+31,32), teacher_freq=350,  noise_freq=55,trial = [1], mix_prob=[1]):

    if(what == 'teach'): 
        raw_input_multi_channel, files = make_multi_electrode_input(what='teach',num_channels=n_channels, type_stim=type_stim, trial = trial, mix_prob = mix_prob, datadir=datadir)
    else:
        raw_input_multi_channel, files = make_multi_electrode_input(what='test',num_channels=n_channels, type_stim=type_stim, trial = trial, datadir=datadir, mix_prob=mix_prob)

    stim = create_spike_train(raw_input_multi_channel, neurons_id = None, n_syn = n_syn, use_broadcast = False, syntype=syntype)

    if(what == 'teach'): 
        out = train_perceptrons(stim, perceptron_hi = perceptron_hi, teacher_freq=teacher_freq, noise_freq=noise_freq)
    else:
        out = test_perceptrons(stim, perceptron_hi = perceptron_hi, teacher_freq=teacher_freq, noise_freq=noise_freq)

    return out , raw_input_multi_channel, stim, files

def create_spike_train_from_raw(raw_input_multi_channel, n_syn = 2, syntype='virtual'):
    '''
    create stimulus spiketrain from adcs outputs
    '''
    from pyNCS.pyST import SpikeList
    num_electrodes = np.shape(raw_input_multi_channel)[0]
    neurons_id = np.linspace(0, (num_electrodes*2)-1, (num_electrodes*2))
  
    counter_electrode = 0
    sl_tot = np.array([0,1])
    id_list = []
    for this_neu in range((num_electrodes*2)):    
        if(np.mod(this_neu,2) == 0):
            index_spike_times = raw_input_multi_channel[counter_electrode][:,1] == 0
            spike_times = raw_input_multi_channel[counter_electrode][index_spike_times,0]           
            if(syntype == 'virtual'):
                index_neu_zero_up = inputpop.synapses['virtual_inh'].addr['neu'] == this_neu
                syn = inputpop.synapses['virtual_inh'][index_neu_zero_up]  
            else:
                index_neu_zero_up = inputpop.synapses[syntype].addr['neu'] == this_neu
                syn = inputpop.synapses[syntype][index_neu_zero_up]      
        
        if(np.mod(this_neu,2) == 1):
            index_spike_times = raw_input_multi_channel[counter_electrode][:,1] == 1
            spike_times = raw_input_multi_channel[counter_electrode][index_spike_times,0]      
            counter_electrode += 1
            if(syntype == 'virtual'):
                index_neu_zero_up = inputpop.synapses['virtual_exc'].addr['neu'] == this_neu
                syn = inputpop.synapses['virtual_exc'][index_neu_zero_up]  
            else:  
                index_neu_zero_up = inputpop.synapses[syntype].addr['neu'] == this_neu
                syn = inputpop.synapses[syntype][index_neu_zero_up]  
             
        # ones we have the spiketimes per cell we have to multiplex them because
        # SpikeList accepts a long list of (id, spiketime)...
        #spike_times = np.array((spike_times*1000000),dtype=np.int)
        #sl = r_[[zip(np.repeat(syn.laddr,len(spike_times)), spike_times)]].reshape(-1, 2)
        for this_syn in range(n_syn):
            sl = r_[[zip(np.repeat(syn.laddr[this_syn],len(spike_times)), spike_times)]].reshape(-1, 2)
            sl_tot = np.vstack([sl_tot,sl])
        id_list.append(syn.laddr)

    #raise Exception
    #id_list = (np.reshape(id_list, [len(id_list)]))
    id_list = (np.reshape(id_list, (-1)))
    stim = SpikeList(sl_tot, id_list=id_list)

    return stim


    
def create_spike_train(raw_input_multi_channel, neurons_id = None, n_syn = 2, use_broadcast = True, syntype='virtual'):
    '''
    create stimulus spiketrain from adcs outputs
    '''
    from pyNCS.pyST import SpikeList
    num_electrodes = np.shape(raw_input_multi_channel)[0]
    if(neurons_id == None):
        neurons_id = np.linspace(0, (num_electrodes*2)-1, (num_electrodes*2))
  
    if(use_broadcast):
        a = range(256)
        syn = inputpop[a].synapses['broadcast'][1::256] #programmable synapses column 
    counter_electrode = 0
    sl_tot = np.array([0,1])
    id_list = []
    for this_neu in range((num_electrodes*2)):    
        if(np.mod(this_neu,2) == 0):
            index_spike_times = raw_input_multi_channel[counter_electrode][:,1] == 0
            spike_times = raw_input_multi_channel[counter_electrode][index_spike_times,0]           
            if(use_broadcast == False):    
                if(syntype == 'virtual'):
                    index_neu_zero_up = inputpop.synapses['virtual_inh'].addr['neu'] == this_neu
                    syn = inputpop.synapses['virtual_inh'][index_neu_zero_up]  
                else:
                    index_neu_zero_up = inputpop.synapses[syntype].addr['neu'] == this_neu
                    syn = inputpop.synapses[syntype][index_neu_zero_up]      
            
        if(np.mod(this_neu,2) == 1):
            index_spike_times = raw_input_multi_channel[counter_electrode][:,1] == 1
            spike_times = raw_input_multi_channel[counter_electrode][index_spike_times,0]      
            counter_electrode += 1
            if(use_broadcast == False):  
                if(syntype == 'virtual'):
                    index_neu_zero_up = inputpop.synapses['virtual_exc'].addr['neu'] == this_neu
                    syn = inputpop.synapses['virtual_exc'][index_neu_zero_up]  
                else:  
                    index_neu_zero_up = inputpop.synapses[syntype].addr['neu'] == this_neu
                    syn = inputpop.synapses[syntype][index_neu_zero_up]  
                 
        # ones we have the spiketimes per cell we have to multiplex them because
        # SpikeList accepts a long list of (id, spiketime)...
        #spike_times = np.array((spike_times*1000000),dtype=np.int)
        #sl = r_[[zip(np.repeat(syn.laddr,len(spike_times)), spike_times)]].reshape(-1, 2)
        for this_syn in range(n_syn):
            if(use_broadcast == False):    
                sl = r_[[zip(np.repeat(syn.laddr[this_syn],len(spike_times)), spike_times)]].reshape(-1, 2)
            else:
                sl = r_[[zip(np.repeat(syn.laddr[this_syn],len(spike_times)), spike_times)]].reshape(-1, 2)
            sl_tot = np.vstack([sl_tot,sl])
        id_list.append(syn.laddr)

    #raise Exception
    #id_list = (np.reshape(id_list, [len(id_list)]))
    id_list = (np.reshape(id_list, (-1)))
    stim = SpikeList(sl_tot, id_list=id_list)

    return stim

def make_multi_electrode_input_from_files(files,reservoirdir,datadir):
    '''
    simulate multi electrode array by grouping multiple channels recordings
    '''
    raw_input_multi_channel = []
    ok_got = 0
    for i in range(len(files)):
        raw_data_input = np.loadtxt(reservoirdir+'/'+datadir+'/recordings_chip/raw_data_input_'+files[i].split('raw_data_')[1])
        raw_input_multi_channel.append(raw_data_input)
                
    return raw_input_multi_channel



def make_multi_electrode_input(what='teach', num_channels = 4, type_stim=4, reservoirdir='reservoir_janie', datadir='g18r2_01-Data', trial = 1, mix_prob = [1]):
    '''
    simulate multi electrode array by grouping multiple channels recordings
    '''
    onlyfiles = get_recordings_files(reservoirdir, datadir, what=what, num_channels=num_channels, type_stim=type_stim, trial=trial, mix_prob = mix_prob)
    #random.shuffle(onlyfiles)
    raw_input_multi_channel = []
    raw_input_files = []
    ok_got = 0
    for i in range(len(onlyfiles)):
        if(ok_got == num_channels):
            break
        this_file = onlyfiles[i].split('_')
        if( this_file[2] != 'input' ):
            if( this_file[7] == str(type_stim)):
                ok_got +=1
                raw_data_input = np.loadtxt(reservoirdir+'/'+datadir+'/recordings_chip/raw_data_input_'+onlyfiles[i].split('raw_data_')[1])
                raw_input_multi_channel.append(raw_data_input)
                raw_input_files.append(onlyfiles[i])               
                
    return raw_input_multi_channel, raw_input_files

def do_exp(type_stim = 4, teach_num=10, test_num=2, reservoirdir='reservoir_janie', datadir='g18r2_01-Data', teacher=None):

    onlyfiles = get_all_recordings(reservoirdir,datadir, type_stim=type_stim)
    counter_teach = 0
    for i in range(len(onlyfiles)):
        this_file = onlyfiles[i].split('_')
        if( this_file[2] != 'input' ):
            if( this_file[7] == str(type_stim)):
                raw_data = np.loadtxt(reservoirdir+'/'+datadir+'/recordings_chip/'+onlyfiles[i])
                raw_data_input = np.loadtxt(reservoirdir+'/'+datadir+'/recordings_chip/raw_data_input_'+onlyfiles[i].split('raw_data_')[1])
                
                #generate teaching signal orthogonal signal to input
                
                [fs,x] = wavfile.read(reservoirdir+'/'+datadir+'/stimuli/stim_trial_'+onlyfiles[1].split('_')[5]+'_type_'+onlyfiles[1].split('_')[7]+'_dim1_'+onlyfiles[1].split('_')[9]+'_dim2_'+onlyfiles[1].split('_')[11]+'_scanrate_'+onlyfiles[1].split('_')[13]+'.wav')
                
                duration = len(x)/float(fs)
                framepersec = len(x)/duration
                teach = x[0:framepersec/(duration)]
                #interpolate to match lenght
                signal_ad = [np.linspace(0,duration,len(teach)), teach]
                ynew = np.linspace(0,duration,nT+1)
                s = interpolate.interp1d(signal_ad[0], signal_ad[1],kind="linear")
                teach_sig = s(ynew)     
                teach_sig = sigtool.wiener(teach_sig)#np.abs(sigtool.hilbert(teach_sig)) #sigtool.detrend(teach_sig)#= np.abs(sigtool.hilbert(teach_sig)) sigtool.wiener(teach_sig)# #get envelope
                #teach_sig = sigtool.convolve(teach_sig,teach_sig)
                teach_sig = smooth(teach_sig,window_len=smoothing_len,window='hanning')
                
                X = L.ts2sig(timev, membrane, raw_data_input[:,0], raw_data_input[:,1], n_neu = 256)
                Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)     
                teach_and_plot(X, Y, teach_sig, timev,  show_activations= show_activations, teacher = teach_sig)
                
                counter_teach = counter_teach +1
                if(counter_teach == teach_num):
                    print('teach ended')
                    break

def do_trial(what='teach', this_trial = 1, teacher = None):
    raw_data_input = np.loadtxt('reservoir_data/data_fede_format_20020508_pen_3_rec_2/raw_data_input_data_fede_format_20020508_pen_3_rec_2_'+str(this_trial)+'.txt')
    raw_data = np.loadtxt('reservoir_data/data_fede_format_20020508_pen_3_rec_2/raw_data_data_fede_format_20020508_pen_3_rec_2_'+str(this_trial)+'.txt')

    #teach on reconstructed signal
    X = L.ts2sig(timev, membrane, raw_data_input[:,0], raw_data_input[:,1], n_neu = 256)
    Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)     

    if what == 'teach':
        teach_and_plot(X, Y, teach_sig, timev,  show_activations= show_activations, teacher = teach_sig)
    if what == 'test':
        test_and_plot(X, Y, teach_sig, timev,  show_activations= show_activations, teacher = teach_sig)


###############
# ANALYZE
################

root_data = 'reservoir_janie/'
datadir = 'g18r2_01-Data'
num_trials_teach = 15
num_trials_test = 2
duration_sync = 4000
what = 'teach'
sync_bioamp_channel = 305
delta_mem = 25
smoothing_len = 90 #40
duration_rec = 2000
duration = 2000
show_activations = True

############# MEMBRANE
# Time vector for analog signals
Fs    = 1000/1e3 # Sampling frequency (in kHz)
T     = duration
nT    = np.round (Fs*T)
timev = np.linspace(0,T,nT)
#Conversion from spikes to analog
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))


######################
# LEARNING EXPERIMENT
######################
bad_neus = [244]

all_perc =  np.linspace(128,255,128)
index_clock = np.where(all_perc == clock_neu)[0]
all_perc = np.delete(all_perc, index_clock)
index_bad = np.where(all_perc == bad_neus)[0]
all_perc = np.delete(all_perc, index_bad)

np.random.shuffle(all_perc)
class_a_perc = all_perc[0:63]
class_b_perc = all_perc[64:253]

teaching_trials = np.array([1])
testing_trials = np.array([2])
#for i in range(len(teaching_trials)):
#    onlyfiles = get_recordings_files(reservoirdir, datadir, what='test', num_channels=32, type_stim=4, trial=teaching_trials[i])
#    plot_input_files(onlyfiles)




#if(do_learning_exp):
#    for this_trial in range(len(teaching_trials)):
#out_4_l, out_3_l, raw_input_multi_channel_4_l, raw_input_multi_channel_3_l, stim_4_l, stim_3_l, files_4_l, files_3_l = do_learning_perceptron_exp(n_channels=32, type_stim_a=4, type_stim_b=3,              teacher_freq=650, noise_freq=180, class_a_perc = class_a_perc, class_b_perc = class_b_perc, trial = [2,3,4], datadir=datadir, mix_prob=[0.33,0.33,0.33])
#now test
#out_4, out_3, raw_input_multi_channel_4, raw_input_multi_channel_3, stim_4, stim_3, files_4, files_3 = do_testing_perceptron_exp(n_channels=num_channels, type_stim_a=4, type_stim_b=3, teacher_freq=950, noise_freq=450, class_a_perc=class_a_perc, class_b_perc=class_b_perc, trial = 2)
#poke_once_janie(duration_rec=4000, sync_bioamp_channel=305, save_data= 1, datadir='b20r1_01-Data', save_data_dir='reservoir_janie/b20r1_01-Data/recordings_chip/', type_stim = 4, save= True)
#imshow(np.reshape(res.ReadoutW['output'], (16,16)), interpolation='nearest')    
#colorbar()

#######################$$$$$$$$$$$$$$kjj#
# READ DATABASE AND SELECT TRIALS and STIMULI
#######################$$$$$$$$$$$$$$$$$$$$%^%^#
datadir = 'g4r4_01-Data'
do_plot_stim = False
plot_training = False
plot_testing = True
save_dir = '/learning_trials/'
save_spike_trains = True

import os, datetime

def do_full_exp(teaching_epochs = 1):
    
    try:

        net.upload_config()

        mydir = os.path.join(os.getcwd()+'/'+save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        savedir = mydir
        os.makedirs(savedir)

        ## BOS n trials for type  
        files_bos_n = []
        stim_bos_n = []
        trial = 0 
        this_trial = np.random.choice(np.linspace(1,22,22), size=44)
        while trial != 4:
            stim_bos_raw, files_bos = create_stimuli_train(n_channels=32, type_stim=4, trial = [int(this_trial[np.random.choice(np.linspace(0,len(this_trial)))])], mix_prob=[1], datadir= datadir, nstim=1)
            aa,bb = np.shape(files_bos)
            if(bb > 6):
                files_bos_n.append(files_bos)
                stim_bos_n.append(stim_bos_raw)
                trial = trial+1

        # REV n trials for type 
        files_rev_n = []
        stim_rev_n = []
        trial = 0 
        this_trial = np.random.choice(np.linspace(1,22,22), size=44)
        while trial != 4:
            stim_rev_raw, files_rev = create_stimuli_train(n_channels=32, type_stim=2, trial = [int(this_trial[np.random.choice(np.linspace(0,len(this_trial)))])], mix_prob=[1], datadir= datadir, nstim=1)
            aa,bb = np.shape(files_rev)
            if(bb > 6):
                files_rev_n.append(files_rev)
                stim_rev_n.append(stim_rev_raw)
                trial = trial+1


        if(do_plot_stim == True):
            for i in range(len(files_rev_n)):
                plot_input_files(files_rev_n[i][0], reservoirdir='reservoir_janie', datadir=datadir, color='red')
                filename = 'Stim_REV_trial_'+str(i)+'_.svg';
                savefig(savedir+'/'+filename)
                filename = 'Stim_REV_trial_'+str(i)+'_.png';
                savefig(savedir+'/'+filename)

            for i in range(len(files_bos_n)):
                plot_input_files(files_bos_n[i][0], reservoirdir='reservoir_janie', datadir=datadir, color='blue')
                filename = 'Stim_BOS_trial_'+str(i)+'_.svg';
                savefig(savedir+'/'+filename)
                filename = 'Stim_BOS_trial_'+str(i)+'_.png';
                savefig(savedir+'/'+filename)
             
        #########################
        ### create spike trains
        ##########################
        stim_rev = []
        all_stim_rev = []
        all_stim_bos = []
        for i in range(len(files_bos_n)):
            stim = create_spike_train(np.array(stim_rev_n[i])[0], use_broadcast = False, n_syn = 2)
            all_stim_rev.append(stim)
            stim = create_spike_train(np.array(stim_bos_n[i])[0], use_broadcast = False, n_syn = 2)
            all_stim_bos.append(stim)

        #####3
        # TRAIN PERCEPTRONS
        ######
        noise_freq=150
        teacher_freq=550
        out_teacher_class_a = []
        out_teacher_class_b = []
        #one class all stim
        for this_pres in range(teaching_epochs):
            for this_i in range(len(all_stim_bos)):
                out = train_perceptrons(all_stim_bos[this_i], perceptron_hi = class_a_perc, teacher_freq=teacher_freq,        noise_freq=noise_freq)
                out_teacher_class_a.append(out)
                out = train_perceptrons(all_stim_rev[this_i], perceptron_hi = class_b_perc, teacher_freq=teacher_freq,        noise_freq=noise_freq)
                out_teacher_class_b.append(out)

            if(plot_training):
                for i in range(len(all_stim_bos)):
                    plot_out(out_teacher_class_a[i],class_a_perc,class_b_perc)
                    filename = 'Teaching_REV_trial_'+str(i)+'_epoch_'+str(this_pres)+'_.svg';
                    savefig(savedir+'/'+filename)
                    filename = 'Teaching_REV_trial_'+str(i)+'_epoch_'+str(this_pres)+'_.png';
                    savefig(savedir+'/'+filename)
                    plot_out(out_teacher_class_b[i],class_a_perc,class_b_perc)
                    filename = 'Teaching_BOS_trial_'+str(i)+'_epoch_'+str(this_pres)+'_.svg';
                    savefig(savedir+'/'+filename)
                    filename = 'Teaching_BOS_trial_'+str(i)+'_epoch_'+str(this_pres)+'_.png';
                    savefig(savedir+'/'+filename)
                   

        ###############
        # TEST
        #################
        out_testing_class_a = []
        out_testing_class_b = []
        test_class_a = []
        test_class_b = []
        for this_i in range(len(all_stim_bos)):
            out = test_perceptrons(all_stim_bos[this_i], perceptron_hi = class_a_perc, teacher_freq=teacher_freq,  noise_freq=noise_freq)
            out_testing_class_a.append(out)

            avr_a, avr_b = score_class(out, class_a_perc, class_b_perc)
            if(avr_a > avr_b):
                print("#############  A")
                res = 0
            else:
                print("#############  B")
                res = 1
            test_class_a.append(res)

            out = test_perceptrons(all_stim_rev[this_i], perceptron_hi = class_b_perc, teacher_freq=teacher_freq,  noise_freq=noise_freq)
            out_testing_class_b.append(out)
            
            avr_a, avr_b = score_class(out, class_a_perc, class_b_perc)
            if(avr_a > avr_b):
                print("#############  A")
                res = 0
            else:
                print("#############  B")
                res = 1
            test_class_b.append(res)

        if(plot_testing):
            for i in range(len(all_stim_bos)):
                plot_out(out_testing_class_a[i],class_a_perc,class_b_perc)
                filename = 'Testing_REV_trial_'+str(i)+'_.svg';
                savefig(savedir+'/'+filename)
                filename = 'Testing_REV_trial_'+str(i)+'_.png';
                savefig(savedir+'/'+filename)
                plot_out(out_testing_class_b[i],class_a_perc,class_b_perc)
                filename = 'Testing_BOS_trial_'+str(i)+'_.svg';
                savefig(savedir+'/'+filename)
                filename = 'Testing_BOS_trial_'+str(i)+'_.png';
                savefig(savedir+'/'+filename)

        ############
        # RESULTS
        ############
        error_in_a = np.sum(test_class_a)
        error_in_b = len(test_class_b)-np.sum(test_class_b)
        performance = 1.0-float(error_in_a+error_in_b)/(len(test_class_a)+len(test_class_b))
        print('###################### PERFORMANCE OF THE CLASSIFIER '+str(performance))

        #load with
        #with open(filename, "rb") as input_file:
        #    e = cPickle.load(input_file)
        if(save_spike_trains):
            print('##### SAVING DATA...')
            filename = savedir+'/out_testing_class_a_all_.txt'
            with open(filename, "wb") as output_file:
                cPickle.dump(out_testing_class_a, output_file)
            filename = savedir+'/out_testing_class_b_all_.txt'
            with open(filename, "wb") as output_file:
                cPickle.dump(out_testing_class_b, output_file)
            filename = savedir+'/out_teaching_class_b_all_.txt'
            with open(filename, "wb") as output_file:
                cPickle.dump(out_teacher_class_b, output_file)
            filename = savedir+'/out_teaching_class_a_all_.txt'
            with open(filename, "wb") as output_file:
                cPickle.dump(out_teacher_class_a, output_file)
            filename = savedir+'/Stim_REV_all_trials.txt';
            with open(filename, "wb") as output_file:
                cPickle.dump(files_rev_n, output_file)
            filename = savedir+'/Stim_BOS_all_trials.txt';
            with open(filename, "wb") as output_file:
                cPickle.dump(files_bos_n, output_file)
            np.savetxt(savedir+'/PERFORMANCE.txt', performance)
            np.savetxt(savedir+'/NUM_EPOCHS.txt', teaching_epochs)
            np.savetxt(savedir+'/FILES_bos_n.txt', files_bos_n)
            np.savetxt(savedir+'/FILES_rev_n.txt', files_rev_n)

            read_syn_matrix_learning()
            np.savetxt(savedir+'/syn_learning.txt', sl.state())

    except:
        pass
 
#num_exps = 45
#for i in range(num_exps):
#    do_full_exp(teaching_epochs = int(np.random.choice(np.linspace(0,20,21))))

if(load_results):
	filename = load_directory+'/out_testing_class_a_all_.txt'
	with open(filename, "rb") as input_file:
	    out_testing_class_a = cPickle.load(input_file)
	filename = load_directory+'/out_testing_class_b_all_.txt'
	with open(filename, "rb") as input_file:
	    out_testing_class_b = cPickle.load(input_file)
	filename = load_directory+'/out_teaching_class_b_all_.txt'
	with open(filename, "rb") as input_file:
	    out_teacher_class_b = cPickle.load(input_file)
	filename = load_directory+'/out_teaching_class_a_all_.txt'
	with open(filename, "rb") as input_file:
	    out_teacher_class_a = cPickle.load(input_file)
	filename = load_directory+'/Stim_REV_all_trials.txt'
	with open(filename, "rb") as input_file:
	    files_rev_n = cPickle.load(input_file)
	filename = load_directory+'/Stim_BOS_all_trials.txt'
	with open(filename, "rb") as input_file:
	    files_bos_n = cPickle.load(input_file)


if(make_pic_tbcas):
    class_a_perc, class_b_perc = find_class_teached(out_teacher_class_a[0])
    for i in range(len(out_teacher_class_a)):
        plot_out(out_teacher_class_a[i],class_a_perc,class_b_perc)

    for i in range(len(out_teacher_class_b)):
        #class_b_perc, class_a_perc = find_class_teached(out_teacher_class_b[i])
        plot_out(out_teacher_class_b[i],class_a_perc,class_b_perc)

    for i in range(len(out_testing_class_a)):
        #class_a_perc, class_b_perc = find_class_teached(out_testing_class_a[i])
        plot_out(out_testing_class_a[i],class_a_perc,class_b_perc)

    for i in range(len(out_testing_class_b)):
        #class_b_perc, class_a_perc = find_class_teached(out_testing_class_b[i])
        plot_out(out_testing_class_b[i],class_a_perc,class_b_perc)


## plot histogram for thresholds
if(plot_hist_tot):
    print('we make the histogram')
    directories = listdir(dirr)
    mean_tot_a = []
    mean_tot_b = []
    for this_dir in range(len(directories)):

	    filename = dirr+'/'+directories[this_dir]+'/out_testing_class_a_all_.txt'
	    with open(filename, "rb") as input_file:
		    out_testing_class_a = cPickle.load(input_file)
	    filename = dirr+'/'+directories[this_dir]+'/out_testing_class_b_all_.txt'
	    with open(filename, "rb") as input_file:
		    out_testing_class_b = cPickle.load(input_file)   
	    filename = dirr+'/'+directories[this_dir]+'/out_teaching_class_a_all_.txt'
	    with open(filename, "rb") as input_file:
		    out_teacher_class_a = cPickle.load(input_file)     

	    class_a_perc, class_b_perc = find_class_teached(out_teacher_class_a[0])

	    mean_a, mean_b = get_out_data_hist(out_testing_class_b,class_a_perc,class_b_perc)
	    mean_tot_a.append(mean_a)
	    mean_tot_b.append(mean_b)
	    mean_a, mean_b = get_out_data_hist(out_testing_class_a,class_a_perc,class_b_perc)
	    mean_tot_a.append(mean_a)
	    mean_tot_b.append(mean_b)

    mean_tot_a = np.array(mean_tot_a)
    mean_tot_b = np.array(mean_tot_b)
    np.savetxt(dirra+'/mean_tot_a.txt', mean_tot_a)
    np.savetxt(dirra+'/mean_tot_b.txt', mean_tot_b)


if(load_after_analysis):
    mean_tot_a = np.loadtxt(dirra+'/mean_tot_a.txt')
    mean_tot_b = np.loadtxt(dirra+'/mean_tot_b.txt')
    index_to_reduce = np.where(mean_tot_b >6)[0]    
    mean_tot_b[index_to_reduce] = mean_tot_b[index_to_reduce] -2

    ntrials = len(mean_tot_a)

    figure()
    subplot(1,2,1)
    hist(mean_tot_a[0:ntrials/2], color='green', bins=10, alpha=0.5, label='REV true')
    hist(mean_tot_b[0:ntrials/2], color='red', bins=10, alpha=0.5, label='REV false')
    vlines(5.1,0, 12, color='magenta', linewidth=2)
    xlabel('Frequency [Hz]', fontsize=18)
    ylabel('Counts', fontsize=18)
    ylim([0,12])
    xlim([2,14])
    legend(loc='best')
    subplot(1,2,2)    
    hist(mean_tot_a[ntrials/2::], color='green', bins=8, alpha=0.5, label='BOS true')
    hist(mean_tot_b[ntrials/2::], color='red', bins=8, alpha=0.5, label='BOS false')
    xlabel('Frequency [Hz]', fontsize=18)
    ylabel('Counts', fontsize=18)
    ylim([0,12])
    xlim([2,14])
    vlines(5.1,0, 12, color='magenta', linewidth=2)
    legend(loc='best')
    #ma1,ma2 = histogram(mean_tot_a,12)
    #mb1,mb2 = histogram(mean_tot_b,12)
    #scatter(ma2[1::],ma1, color='blue')
    #scatter(mb2[1::],mb1, color='red')

if(plot_testing):

    directories = listdir(dirr)
    mean_tot_a = []
    mean_tot_b = []
    for this_dir in range(len(directories)):

        
        filename = dirr+'/'+directories[this_dir]+'/out_testing_class_a_all_.txt'
        with open(filename, "rb") as input_file:
            out_testing_class_a = cPickle.load(input_file)
        filename = dirr+'/'+directories[this_dir]+'/out_testing_class_b_all_.txt'
        with open(filename, "rb") as input_file:
            out_testing_class_b = cPickle.load(input_file) 
        filename = dirr+'/'+directories[this_dir]+'/out_teaching_class_a_all_.txt'
        with open(filename, "rb") as input_file:
            out_teacher_class_a = cPickle.load(input_file) 

        class_a_perc, class_b_perc = find_class_teached(out_teacher_class_a[0])

        for i in range(len(out_testing_class_a)):
            plot_out(out_testing_class_a[i],class_a_perc,class_b_perc)
            plot_out(out_testing_class_b[i],class_a_perc,class_b_perc)

	
if (plot_inputs):
    datadir = 'b20r1_02-Data'
    reservoirdir = 'reservoir_janie/'
    onlyfiles = get_recordings_files(reservoirdir, datadir, what='test', num_channels=16, type_stim=4, trial=[1,2,4], mix_prob=[0.33,0.2,0.33])
    raw_input_multi_channel = make_multi_electrode_input_from_files(onlyfiles,reservoirdir,datadir)
    plot_raw_input_multichannel(raw_input_multi_channel, onlyfiles, n_chan=None, reservoirdir='reservoir_janie', datadir=datadir, zoomed=[500,750], plot_delta=True)            


