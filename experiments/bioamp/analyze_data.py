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

sys.path.append('/home/federico/projects/work/api/reservoir/')

from scipy import interpolate
import reservoir as L

ion()

load_config = True
config_dir = 'data_reservoir/ee03-ii02_b4/'
wavefile =  'Hackenelektrodenableitung_mecopoda_elongata_chirper_2.wav'

################
# RESERVOIR
################
res = L.Reservoir()
if(load_config == True):
    res.load_config(config_dir)
res.reset(alpha=np.logspace(-7,-6,100))

#rhoW = max(abs(linalg.eig(res.matrix_programmable_w)[0]))
#scaled_matrix_programmable_w *= 1.25 / rhoW   #no meaning

#res.reset(alpha=np.logspace(-8,8,50))
#res.reset(alpha=np.logspace(-10,10,500)) #init with right alphas

###############
# FUNCTIONS
###############

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

root_data = 'reservoir_data/'
datadir_1 = 'data_fede_format_20020508_pen_3_rec_2'
datadir_2 = 'data_fede_format_20020608_pen_2_rec_2'
num_trials_teach = 15
num_trials_test = 2
duration_sync = 4000
datadir = datadir_2
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

###########################
# PRODUCE TEACHER SIGNAL
###########################

#generate teaching signal orthogonal signal to input
teach = np.loadtxt('/home/federico/project/work/trunk/data/Berkeley/rad-auditory/wehr/Tools/'+datadir+'/1_.txt')#
framepersec = len(teach)/15
teach = teach[0:framepersec/(duration_rec/1000)]
#interpolate to match lenght
signal_ad = [np.linspace(0,duration_rec,len(teach)), teach]
ynew = np.linspace(0,duration_rec,nT+1)
s = interpolate.interp1d(signal_ad[0], signal_ad[1],kind="linear")
teach_sig = s(ynew)     
teach_sig = sigtool.wiener(teach_sig)#np.abs(sigtool.hilbert(teach_sig)) #sigtool.detrend(teach_sig)#= np.abs(sigtool.hilbert(teach_sig)) sigtool.wiener(teach_sig)# #get envelope
#teach_sig = sigtool.convolve(teach_sig,teach_sig)
teach_sig = smooth(teach_sig,window_len=smoothing_len,window='hanning')


#######################
# TEACH AND TEST
######################

for this_trial in range(1,num_trials_test+num_trials_teach):
    if this_trial <= num_trials_teach-1:
        do_trial(what='teach', this_trial = this_trial, teacher = teach_sig)
    if this_trial > num_trials_teach:
        do_trial(what='test', this_trial = this_trial, teacher = teach_sig)


#imshow(np.reshape(res.ReadoutW['output'], (16,16)), interpolation='nearest')    
#colorbar()

