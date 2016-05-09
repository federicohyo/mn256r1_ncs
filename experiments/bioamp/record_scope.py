import pyAgilent
from pylab import *
import matplotlib
import numpy as np
import sys
sys.path.append('/home/federico/projects/work/trunk/code/python/spkInt/scripts/')
import functions

#init oscilloscope
osc = pyAgilent.Agilent(host="172.19.10.159");
#osc._send_command('WAV:FORM ASC');
#osc._send_command('WAV:POIN:MODE RAW');
#osc._send_command('WAV:POINTS 2000000');

osc.engage()
#data = osc.read_data([1,2,3,4]);

up = osc._read_data_from_channel(2)
down = osc._read_data_from_channel(4)
amp = osc._read_data_from_channel(3)
gout = osc._read_data_from_channel(1)

maxs,mins = functions.peakdet(gout,0.8)
maxs = np.array(maxs)
spiketrain_pulse = [maxs[:,0],np.repeat(1,len(maxs))]

def db(x):
    """Convert the specified power value to decibels assuming a
    reference value of 1."""
    return 10*log10(x)

def snr(u, u_rec, k_min=0, k_max=None):
    """Compute the signal-to-noise ratio (in dB) of a signal given its
    reconstruction.
    Parameters
    ----------
    u : numpy array of floats
    Original signal.
    u_rec : numpy array of floats
    Reconstructed signal.
    k_min : int
    Lower index into the signal over which to compute the SNR.
    k_max : int
    Upper index into the signal over which to compute the SNR.
    """
    if len(u) != len(u_rec):
        raise ValueError('u and u_rec must be the same length')
    return db(mean(u[k_min:k_max]**2))-db(mean((u[k_min:k_max]-u_rec[k_min:k_max])**2))
    
##### Mean firing rate of each neuron
def meanNeuFiring(SpikeTrain, n_neurons,simulation_time):
	ArraySpike = np.array(SpikeTrain)
	MeanRate   = np.zeros([len(n_neurons)])
	for i in range(len(n_neurons)):
		Index_n = np.where(np.logical_and(ArraySpike[1] == n_neurons[i], np.logical_and(ArraySpike[0] > simulation_time[0] , ArraySpike[0] < simulation_time[1] )) )
		MeanRate[i] = len(Index_n[0])*1000.0/(simulation_time[1]-simulation_time[0]) # time unit: ms
	return MeanRate

def meanratesignal(SpikeTrain, simulation_time):
    ArraySpike = np.array(SpikeTrain)
    MeanRate   = np.zeros([1])
    Index_n = np.where(np.logical_and(ArraySpike > simulation_time[0] , ArraySpike < simulation_time[1] ))
    MeanRate = len(Index_n)*1000.0/(simulation_time[1]-simulation_time[0]) # time unit: ms
    return MeanRate

bins = 150
max_time = max(spiketrain_pulse[0])
edges = np.floor(max_time/bins)

mean = []
for i in range(int(edges)):
	time_start = i*edges
	time_stop = (i+1)*edges 
	tmp_mean = meanNeuFiring(spiketrain_pulse,[1],[time_start,time_stop])
	mean.append(tmp_mean)

time_up = np.linspace(0,10,len(up))
time_down = np.linspace(0,10,len(down))
time_amp = np.linspace(0,10,len(amp))
time_gout = np.linspace(0,10,len(gout))

subplot(4,1,1)
plot(time_gout,gout,'g', label='Input')
xlabel('Time [ms]')
ylabel('V')
legend(loc='best')
subplot(4,1,2)
plot(time_up,up,'r', label='Spike Detector')
xlabel('Time [ms]')
ylabel('V')
legend(loc='best')

subplot(3,1,1)
plot(time_amp,amp,'b', label='Input')
xlabel('Time [us]')
ylabel('V')
legend(loc='best')
subplot(3,1,2)
plot(time_down,down,'m', label='LPF')
xlabel('Time [us]')
ylabel('V')
legend(loc='best')
subplot(3,1,3)
plot(time_gout,gout,'g', label='Pulse Ouput')
xlabel('Time [us]')
ylabel('V')
legend(loc='best')



filename = 'figures/ad_scope_data_features.eps'
savefig(filename,format='eps')
filename = 'figures/ad_scope_data_feature.png'
savefig(filename,format='png')

np.savetxt('data/up_feature.txt', up)
np.savetxt('data/down_feature.txt', down)
np.savetxt('data/amp_feature.txt', amp)
np.savetxt('data/gout_feature.txt', gout)
np.savetxt('data/time_up_feature.txt', time_up)
np.savetxt('data/time_down_feature.txt', time_down)
np.savetxt('data/time_amp_feature.txt', time_amp)
np.savetxt('data/time_gout_feature.txt', time_gout)

up = np.loadtxt('data/up.txt')
down = np.loadtxt('data/down.txt')
amp = np.loadtxt('data/amp.txt')
gout = np.loadtxt('data/gout.txt')
time_up = np.loadtxt('data/time_up.txt')
time_down = np.loadtxt('data/time_down.txt')
time_amp = np.loadtxt('data/time_amp.txt')
time_gout = np.loadtxt('data/time_gout.txt')


#3-4
filtered = osc._read_data_from_channel(4)
inp = osc._read_data_from_channel(3)
time_filtered = np.linspace(0,50,len(filtered))
time_inp = np.linspace(0,50,len(inp))

np.savetxt('band-pass/time_inp.txt', time_inp )
np.savetxt('band-pass/inp.txt', inp)
np.savetxt('band-pass/filtered.txt', filtered)
np.savetxt('band-pass/time_filtered.txt', time_filtered)

plot(time_inp,inp-0.2,'g', label='Input')
plot(time_filtered,filtered*20,'r', label='Filtered')
xlabel('Time [ms]')
ylabel('V')
legend(loc='best')

filename = 'band-pass/plot_1.eps'
savefig(filename,format='eps')
filename = 'band-pass/plot_1.png'
savefig(filename,format='png')

######################################
#### reconstruction of neural signal
######################################
up = osc._read_data_from_channel(1)
down = osc._read_data_from_channel(2)
amp = osc._read_data_from_channel(4)
gout = osc._read_data_from_channel(3)

time_up = np.linspace(0,10,len(up))
time_down = np.linspace(0,10,len(down))
time_amp = np.linspace(0,10,len(amp))
time_gout = np.linspace(0,10,len(gout))

np.savetxt('time_up.txt', time_up )
np.savetxt('time_down.txt', time_down )
np.savetxt('time_amp.txt', time_amp )
np.savetxt('time_gout.txt', time_gout )
np.savetxt('up.txt', up )
np.savetxt('down.txt', down )
np.savetxt('amp.txt', amp )
np.savetxt('gout.txt', gout )




time_up = np.loadtxt('time_up.txt' )
time_down  = np.loadtxt('time_down.txt')
time_amp = np.loadtxt('time_amp.txt' )
time_gout  = np.loadtxt('time_gout.txt')
up = np.loadtxt('up.txt')
down = np.loadtxt('down.txt')
amp = np.loadtxt('amp.txt')
gout = np.loadtxt('gout.txt')

figure()
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
#subplot(3,1,1)
#plot(time_amp,amp,'b', label='lna output')
#xlabel('Time [ms]')
#ylabel('V')
#legend(loc='best')
subplot(3,1,2)
plot(time_gout,gout,'g', label='Gout')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(3,1,3)
vlines(time_up[up_min][:,0],0,0.5, 'b', label='UP')
vlines(time_down[dn_min][:,0],0,-0.5, 'r', label='DN')
xlabel('Time [ms]', fontsize=18)
ylabel('Amp [V]', fontsize=18)
legend(loc='best')



up_max, up_min = functions.peakdet(up,1.5)
dn_max, dn_min = functions.peakdet(down,1.5)
figure()
plot(time_up[up_min],up[up_min], 'bd')
plot(time_up,up, 'r', label='UP')
plot(time_down[dn_min],down[dn_min], 'gd')
plot(time_down,down, 'g', label='UP')

signal = np.zeros(len(up))
signal[0] = 0
delta_up = 0.07
delta_dn = 0.095
counter_up = 0
counter_dn = 0
do = 0
for i in range(1,len(up)-1):
    if(counter_up != len(up_min)):
        if(up_min[counter_up][0] == i):
            signal[i] = signal[i-1] + delta_dn
            counter_up = counter_up + 1
            do = 1
    if(counter_dn != len(dn_min)):        
        if(dn_min[counter_dn][0] == i):    
            signal[i] = signal[i-1] - delta_up
            counter_dn = counter_dn + 1
            do = 1
    if(do == 0):
        signal[i] = signal[i-1]
    do = 0

subplot(3,1,1)
plot(time_amp, norm_amp, 'b', label = 'LNA output')
ylabel('Amp [V]', fontsize=18)
plot(time_amp, (norm_signal-np.mean(norm_signal))*1.45, 'r', label='Reconstructed')
legend(loc='best')
ylim([-1,1.1])

figure()
#norm_signal = signal/np.max(signal)
signal = 0.8-signal
norm_signal = signal/np.max(signal)
norm_amp = amp/np.max(amp)
ss = (norm_signal-np.mean(norm_signal))
plot(time_amp, norm_amp, 'b', label = 'LNA output')
plot(time_amp, (norm_signal-np.mean(norm_signal))*1.45, 'r', label='Reconstructed')

print('Signal-to-noise error ratio')
print snr(norm_amp,ss*1.45)

##################################
## PULSE ADC
##################################
from sklearn.preprocessing import scale

pulse = osc._read_data_from_channel(1)
input_pulse = osc._read_data_from_channel(4)

time_pulse = np.linspace(0,10,len(pulse))
time_input_pulse = np.linspace(0,10,len(input_pulse))

np.savetxt('pulse.txt', pulse)
np.savetxt('input_pulse.txt', input_pulse)
np.savetxt('time_pulse.txt', time_pulse)
np.savetxt('time_input_pulse.txt', time_input_pulse)


maxs,mins = functions.peakdet(pulse,3.0)
maxs = np.array(maxs)
mins = np.array(mins) 
mins = map(int,mins[:,0])
maxs = map(int,maxs[:,0])

t_bin = 0.04
t_max = np.max(time_pulse[maxs])
t_min = 0
n_slices = np.floor((t_max-t_min)/t_bin) 
edges = np.linspace(t_min,t_max,n_slices)

mean_rates = []
for i in range(len(edges)-1):
    Index_n = np.where(np.logical_and(time_pulse[maxs] >= edges[i] , time_pulse[maxs] < edges[i+1] ))
    MeanRate = len(Index_n[0])*1000.0/(edges[i+1]-edges[i]) # time unit: ms
    mean_rates.append(MeanRate)
mean_rates = np.array(mean_rates)

input_pulse_n = scale(input_pulse, axis=0, with_mean=True, with_std=True, copy=True )
mean_rates_n = scale(mean_rates, axis=0, with_mean=True, with_std=True, copy=True )
mean_rates_n = mean_rates_n+2/np.max(mean_rates_n+2)
input_pulse_n = input_pulse_n+2/np.max(input_pulse_n+2)

fig, ax1 = subplots()
subplot(3,1,1)
plot(time_input_pulse,input_pulse,'b-', label='Input')
xlim([0,10])
legend(loc='best')
ylabel('Amp [V]', fontsize=18)
subplot(3,1,2)
vlines(time_pulse[maxs],300,600, 'b', linewidth=0.1, label='Pulse ADC')
xlim([0,10])
legend(loc='best')
ylabel('Freq [kHz]', fontsize=18)
subplot(3,1,3)
plot(time_input_pulse,input_pulse_n,'b-', label='Input')
xlim([0,10])
plot(edges[1::],mean_rates_n, 'r--' , linewidth=2, label='Reconstructed')
legend(loc='best')
ylabel('A.U.', fontsize=18)
xlabel('Time [ms]', fontsize=18)
subplot(3,1,2)
ax2 = ax1.twinx()
subplot(3,1,2)
plot(edges[1::],mean_rates/1000, 'r-' , linewidth=2, label='Reconstructed')
ax2.set_ylabel('Freq [kHz]', color='r')
