import pyAgilent
from pylab import *
import matplotlib
import numpy as np
import sys

#init oscilloscope
osc = pyAgilent.Agilent(host="172.19.10.159");
#osc._send_command('WAV:FORM ASC');
#osc._send_command('WAV:POIN:MODE RAW');
#osc._send_command('WAV:POINTS 2000000');

osc.engage()

up = osc._read_data_from_channel(2)
dn = osc._read_data_from_channel(1)
amp = osc._read_data_from_channel(4)
gout = osc._read_data_from_channel(3)

time_up = np.linspace(0,5,len(up))
time_dn = np.linspace(0,5,len(dn))
time_amp = np.linspace(0,5,len(amp))
time_gout = np.linspace(0,5,len(gout))

np.savetxt('sine_100/time_up.txt', time_up )
np.savetxt('sine_100/up.txt', up)
np.savetxt('sine_100/time_dn.txt', time_dn )
np.savetxt('sine_100/dn.txt', dn)
np.savetxt('sine_100/time_amp.txt', time_amp )
np.savetxt('sine_100/amp.txt', amp)
np.savetxt('sine_100/time_gout.txt', time_gout )
np.savetxt('sine_100/gout.txt', gout)

np.savetxt('limit/time_up.txt', time_up )
np.savetxt('limit/up.txt', up)
np.savetxt('limit/time_dn.txt', time_dn )
np.savetxt('limit/dn.txt', dn)
np.savetxt('limit/time_amp.txt', time_amp )
np.savetxt('limit/amp.txt', amp)
np.savetxt('limit/time_gout.txt', time_gout )
np.savetxt('limit/gout.txt', gout)

###########################
### PRODUCE PLOT
###########################

up = np.loadtxt('limit/up.txt')
time_up = np.loadtxt('limit/time_up.txt')
dn = np.loadtxt('limit/dn.txt')
time_dn = np.loadtxt('limit/time_dn.txt')
lna = np.loadtxt('limit/amp.txt')
time_lna = np.loadtxt('limit/time_amp.txt')
gout = np.loadtxt('limit/gout.txt')
time_gout = np.loadtxt('limit/time_gout.txt')



figure()
subplot(4,3,3)
plot(time_lna, lna, color='b', label='Input 1 MHz')
xlim([0,4])
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(4,3,6)
plot(time_gout,gout-np.mean(gout),color='g',label='Gout')
xlim([0,4])
ylim([-0.2,0.2])
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(4,3,9)
plot(time_up,up,color='b',label='UP')
xlim([0,4])
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(4,3,12)
plot(time_dn,dn,color='r',label='DN')
xlim([0,4])
ylabel('Amp [V]', fontsize=18)
xlabel('Time [uS]', fontsize=18)
legend(loc='best')



up = np.loadtxt('sine_100/up.txt')
time_up = np.loadtxt('sine_100/time_up.txt')
dn = np.loadtxt('sine_100/dn.txt')
time_dn = np.loadtxt('sine_100/time_dn.txt')
lna = np.loadtxt('sine_100/amp.txt')
time_lna = np.loadtxt('sine_100/time_amp.txt')
gout = np.loadtxt('sine_100/gout.txt')
time_gout = np.loadtxt('sine_100/time_gout.txt')


subplot(4,3,2)
plot(time_lna,lna-np.mean(lna),color='b', label='Input 100 Hz')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(4,3,5)
plot(time_gout,gout,color='g', label='Gout')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(4,3,8)
plot(time_up,up, color='b',label='UP')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(4,3,11)
plot(time_dn,dn,color='r', label='DN')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
xlabel('Time [ms]', fontsize=18)



up = np.loadtxt('sine_fast_4/up.txt')
time_up = np.loadtxt('sine_fast_4/up_time.txt')
dn = np.loadtxt('sine_fast_4/dn.txt')
time_dn = np.loadtxt('sine_fast_4/dn_time.txt')
lna = np.loadtxt('sine_fast_4/lna.txt')
time_lna = np.loadtxt('sine_fast_4/lna_time.txt')
gout = np.loadtxt('sine_fast_4/gout.txt')
time_gout = np.loadtxt('sine_fast_4/gout_time.txt')

import sys
sys.path.append('/home/federico/projects/work/trunk/code/python/spkInt/scripts/')
import functions

up_max, up_min = functions.peakdet(up,1.6)
dn_max, dn_min = functions.peakdet(dn,1.6)
dn_min = np.array(dn_min)
up_min = np.array(up_min) 
dn_min = map(int,dn_min[:,0])
up_min = map(int,up_min[:,0])

figure()
subplot(3,1,2)
plot(time_gout,gout,color='g', label='Gout')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(3,1,3)
vlines(time_up[up_min],0,0.5, 'b', label='UP')
vlines(time_dn[dn_min],0,-0.5, 'r', label='DN')
ylabel('Amp [V]', fontsize=18)
xlabel('Time [ms]', fontsize=18)
legend(loc='best')

signal = np.zeros(len(up))
signal[0] = -0.12
delta_up = 0.11
delta_dn = 0.11
counter_up = 0
counter_dn = 0
do = 0
for i in range(1,len(up)-1):
    if(counter_up != len(up_min)):
        if(up_min[counter_up] == i):
            signal[i] = signal[i-1] + delta_dn
            counter_up = counter_up + 1
            do = 1
    if(counter_dn != len(dn_min)):        
        if(dn_min[counter_dn] == i):    
            signal[i] = signal[i-1] - delta_up
            counter_dn = counter_dn + 1
            do = 1
    if(do == 0):
        signal[i] = signal[i-1]
    do = 0


subplot(3,1,1)
plot(time_lna,lna-np.mean(lna),color='b', label='Input 3 Hz')
plot(time_up,signal,color='r', label='Reconstructed')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')

plot(time_up,up, color='b',label='UP')

subplot(3,1,3)
plot(time_dn,dn,color='r', label='DN')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
xlabel('Time [ms]', fontsize=18)



up = np.loadtxt('sine_fast_4/up.txt')
time_up = np.loadtxt('sine_fast_4/up_time.txt')
dn = np.loadtxt('sine_fast_4/dn.txt')
time_dn = np.loadtxt('sine_fast_4/dn_time.txt')
lna = np.loadtxt('sine_fast_4/lna.txt')
time_lna = np.loadtxt('sine_fast_4/lna_time.txt')
gout = np.loadtxt('sine_fast_4/gout.txt')
time_gout = np.loadtxt('sine_fast_4/gout_time.txt')
figure()
subplot(3,1,1)
plot(time_lna,lna-np.mean(lna),color='b', label='Input 3 Hz')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(3,1,2)
plot(time_gout,gout,color='g', label='Gout')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(3,1,3)
plot(time_up,up, color='b',label='UP')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
subplot(3,1,3)
plot(time_dn,dn,color='r', linestyle='dotted', label='DN')
ylabel('Amp [V]', fontsize=18)
legend(loc='best')
xlabel('Time [ms]', fontsize=18)


figure()
#freq = 100Hz
#delta = 100mV
#refr = 0.127 n
a = np.array([50,100,200,300])
b = np.array([23,39,63,87])
m,d = polyfit(a, b, 1) 
plot(a, b, 'rd-', a, m*a+d, '--k', label='delta = 100mV') 
xlabel('Input amplitude [mV]', fontsize=18)
ylabel('number of pulses per half cycle', fontsize=18)

b = np.array([43,62,89,120])
m,d = polyfit(a, b, 1) 
plot(a, b, 'yo-', a, m*a+d, '--k', label='delta = 50mV') 
xlabel('Input amplitude [mV]', fontsize=18)
ylabel('number of pulses per half cycle', fontsize=18)
xlim([0,350])
legend(loc='best')

#delta = 20 mV
b = np.array([83,105,164,220])
m,d = polyfit(a, b, 1) 
plot(a, b, 'bx-', a, m*a+d, '--k', label='delta = 20mV') 
xlabel('Input amplitude [mV]', fontsize=18)
ylabel('number of pulses per half cycle', fontsize=18)
xlim([35,325])
legend(loc='best')


#SNR over freqs

######## pulse adc




