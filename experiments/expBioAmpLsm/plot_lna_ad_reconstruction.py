
import numpy as np
import matplotlib
from pylab import *
#np.savetxt('bioamp/time_membrane_lna.txt', time)
#np.savetxt('bioamp/membrane_membrane_lna.txt', membrane)
#np.savetxt('bioamp/index_f.txt', index_f)
#np.savetxt('bioamp/signal_all_0.txt', signal[:,0])
#np.savetxt('bioamp/signal_all_1.txt', signal[:,1])

time = np.loadtxt('bioamp/time_membrane_lna.txt')
membrane = np.loadtxt('bioamp/membrane_membrane_lna.txt')
index_f = np.loadtxt('bioamp/index_f.txt')
index_f = (index_f == 1)
signal = np.zeros([2,22016])
adfs = np.loadtxt('bioamp/signal_all_0.txt')
adfss = np.loadtxt('bioamp/signal_all_1.txt')
signal = np.array([adfs,adfss])
signal = signal.transpose()

figure()
plot(time,membrane*4.2-np.mean(membrane)*4.2, label='LNA output')
plot(signal[index_f,0],signal[index_f,1]-np.mean(signal[index_f,1]),'o-', label='Async. Delta Reconstruction')
ylim([-3,3])
xlim([1140,1159])
xlabel('Time [ms]')
yticks(np.linspace(-3,3,7), np.linspace(0.2,1.5,6))
ylabel('Amp [V]')
legend(loc='best')

times_f = signal[index_f,0]
amps_f = signal[index_f,1]
up = []
dn = []
for i in range(1,len(amps_f)):
    if( amps_f[i] < amps_f[i-1]):
        dn.append(times_f[i])
    else:
        up.append(times_f[i])

up = np.array(up)
dn = np.array(dn)
figure()
plot(times_f,amps_f-np.mean(amps_f), 'ro-', markersize=3,  label='Async. Delta Reconstruction')
vlines(up, 2.65, 2.75)      
vlines(dn, 2.75, 2.85)          
plot(time,membrane*4.2-np.mean(membrane)*4.2, 'g-', label='LNA output')
ylim([-3,3])
xlim([1140,1159])
xlabel('Time [ms]')
yticks(np.linspace(-3,3,7), np.linspace(0.2,1.5,6))
ylabel('Amp [V]')
legend(loc='best')


import scipy
from scipy import stats

signal_ad = [times_f,amps_f-np.mean(amps_f)]
signal_lna = [time,membrane*4.2-np.mean(membrane)*4.2]
sn_ad = scipy.stats.signaltonoise(signal_ad)
sn_lna = scipy.stats.signaltonoise(signal_lna)

import sys
sys.path.append('/home/federico/projects/work/trunk/code/python/spkInt/scripts/')
from scipy import interpolate
import signal_extras 
tck = interpolate.splrep(signal_ad[0], signal_ad[1], s=0)

s = interpolate.interp1d(signal_ad[0], signal_ad[1],kind="linear")
time_n  = time[1900::]
ynew = s(time_n)#interpolate.splev(xnew, tck, der=0)
plt.plot(signal_lna[0], signal_lna[1])
plt.plot(signal_ad[0], signal_ad[1])
plot(time_n, ynew, 'g')

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
 
rmqe =    rmse( ynew ,signal_lna[1][1900::])
print "RMSE", rmqe
figure()
plot(time_n,ynew)
plot(time_n,signal_lna[1][1900::])

print "SNR", signal_extras.snr(signal_lna[1][1900::],ynew)



