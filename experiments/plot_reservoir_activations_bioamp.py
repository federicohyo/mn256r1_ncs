import numpy as np
import matplotlib
from pylab import *
ion()
import wave
import sys
sys.path.append('../api/lsm/')
import lsm as L

liquid = L.Lsm()

spf = wave.open('/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/Hackenelektrodenableitung_mecopoda_elongata_chirper_2_trim.wav','r')
#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
time_orig = np.loadtxt("bioamp/time_orig.txt")
signal = np.loadtxt("bioamp/signal_YM.txt")
spikes_amp = np.loadtxt("bioamp/YM_spikes_amp.txt")
spikes_times = np.loadtxt("bioamp/YM_spikes_times.txt")
Y = np.loadtxt("bioamp/YM.txt")

figure()
plot(time_orig,signal)

index_non_zeros = []
for i in range(256):
    if(np.sum(Y[:,i]) != 0):
        index_non_zeros.append(i)  
size_f = np.floor(np.sqrt(len(index_non_zeros)))

figure()
for i in range(int(size_f**2)):
    #subplot(size_f,size_f,i) 
    plot(Y[:,index_non_zeros[i]])  
    #axis('off')  

figure()
plot(spikes_times, spikes_amp, '*', markersize=2)
ylim([0,256])
xlim([0,6000])
xlabel('Time [ms]')
ylabel('Neu Id')

raw_data_times = np.loadtxt("bioamp/raw_data_times_x.txt")
raw_data_amp = np.loadtxt("bioamp/raw_data_amp_x.txt")

index_ord =  np.argsort(raw_data_times)   
signal = np.zeros([len(raw_data_input)])
for i in range(1,len(raw_data_input)):
    if(raw_data_amp[index_ord[i]] == 1):
        signal[i] = signal[i-1] + delta_up
    if(raw_data_amp[index_ord[i]] == 0):
        signal[i] = signal[i-1] - delta_dn

figure()
signal_trace_good = [raw_data_times[index_ord],signal]                    
plot(signal_trace_good[0], signal_trace_good[1])   
                    
#X = L.ts2sig(timev, membrane, raw_data_times, raw_data_amp, n_neu = 256)
#X = np.loadtxt("bioamp/XXspikes.txt")

# Time vector for analog signals
Fs    = 1000/1e3 # Sampling frequency (in kHz)
T     = 12000
nT    = np.round (Fs*T)
timev = np.linspace(0,T,nT)

#Conversion from spikes to analog
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*1**2)))
X = L.ts2sig(timev, membrane, raw_data_times, raw_data_amp, n_neu = 256)
       
figure()
for i in range(2):
    plot(X[:,i])
    

#SVD
figure()
ac=np.mean(Y**2,axis=0)
aci=np.mean(X**2,axis=0)
max_pos = np.where(ac == np.max(ac))[0]
max_posi = np.where(aci == np.max(aci))[0]
subplot(3,1,1)
plot(X[:,max_posi])
subplot(3,1,2)
plot(Y[:,max_pos])
subplot(3,1,3)
CO = np.dot(Y.T,Y)
CI = np.dot(X.T,X)
si = np.linalg.svd(CI, full_matrices=False, compute_uv=False)
so = np.linalg.svd(CO, full_matrices=True, compute_uv=False)
semilogy(so/so[0], 'bo-', label="outputs")
semilogy(si/si[0], 'go-', label="inputs")
legend(loc="best")   
