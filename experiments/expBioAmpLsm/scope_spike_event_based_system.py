import pyAgilent
from pylab import *
import matplotlib
import sys
sys.path.append('/home/federico/projects/work/trunk/code/python/spkInt/scripts/')
import functions

record = True 


if record == True:
    #init oscilloscope
    osc = pyAgilent.Agilent(host="172.19.10.159");
    osc._send_command('WAV:FORM asc');

    membrane = osc._read_data_from_channel(1)
    lpf = osc._read_data_from_channel(4)
    gout = osc._read_data_from_channel(2)
    pulseout = osc._read_data_from_channel(3)

    np.savetxt('bioamp/gout.txt', gout)
    np.savetxt('bioamp/membrane.txt', membrane)
    np.savetxt('bioamp/pulseout.txt', pulseout)
    np.savetxt('bioamp/lpf.txt', lpf)    
    
else:

    gout = np.loadtxt('bioamp/gout.txt')
    membrane = np.loadtxt('bioamp/membrane.txt')
    pulseout = np.loadtxt('bioamp/pulseout.txt')
    lpf = np.loadtxt('bioamp/lpf.txt')


### from up dn
membrane = osc._read_data_from_channel(1)
up = osc._read_data_from_channel(3)
dn = osc._read_data_from_channel(4)


        
##################################



mm = 0.013
maxs,mins = functions.peakdet(gout,mm)
maxs = np.array(maxs)
mins = np.array(mins)
spiketrain_pulse = [maxs[:,0],np.repeat(1,len(maxs))]

real_maxs_index = np.where(maxs[:,1]>0.9)[0]
maxs = maxs[real_maxs_index,:]
real_min_index = np.where(mins[:,1]<0.8)[0]
mins = mins[real_min_index,:]

mul_fact = len(maxs)/float(len(mins))

## IT DOES NOT WORK
signal = np.zeros([len(maxs)+len(mins),2])
counter_max = 0
counter_min = 0
over_max = False
over_min = False
for i in range(len(maxs)+len(mins)):
    if(not(over_max) and not(over_min)):
        if(maxs[counter_max,0] < mins[counter_min,0]):
            signal[i,:] = [1,maxs[counter_max,0]]
            counter_max = counter_max + 1
            if(counter_max == len(maxs)):
                counter_max = counter_max - 1
                over_min = True
        elif(maxs[counter_max,0] >  mins[counter_min,0]):
            signal[i,:] = [-1,mins[counter_min,0]] 
            counter_min = counter_min + 1
            if(counter_min == len(mins)):
                counter_min = counter_min - 1 
                over_max = True
    if(not(over_max) and over_min):     
        signal[i,:] = [-1,mins[counter_min,0]] 
        counter_min = counter_min + 1
        if(counter_min == len(mins)):
            counter_min = counter_min - 1 
            over_max = True
    if(over_max and not(over_min)):
        signal[i,:] = [1,maxs[counter_max,0]]
        counter_max = counter_max + 1
        if(counter_max == len(maxs)):
            counter_max = counter_max - 1
            over_min = True         

signal[0] = 0.15
deltaup = 0.0047
deltadn = 0.0085
for i in range(len(signal)-1):
    if(signal[i+1,0] == -1):
        signal[i+1,0] = signal[i,0]-deltadn
    else:
        signal[i+1,0] = signal[i,0]+deltaup

#signal = signal[0:200,:]
time_max = np.max(signal[:,1])
time_min = np.min(signal[:,1])
time_signal = np.linspace(np.min(signal[:,1]),np.max(signal[:,1]),len(membrane))

#time_max = 1.2
time = np.linspace(time_min,time_max,len(membrane))
time_lpf = np.linspace(time_min,time_max,len(lpf))
time_gout = np.linspace(time_min,time_max,len(gout))
time_pulseout = np.linspace(time_min,time_max,len(pulseout))


#Selective median filter function for removing oscilloscope artefacts
#(from http://farhan.org/detecting-and-fixing-outliers-in-data.html )
def selective_median_filter(data, kernel=31, threshold=2):
    """Return copy of data with outliers set to median of specified
        window. Outliers are values that fall out of the 'threshold'
        standard deviations of the window median"""
    if kernel % 2 == 0:
        raise Exception("Kernel needs to be odd.")
    n = len(data)
    res = list(data)
    for i in range(0, n):
        seg = res[max(0,i-(kernel/2)):min(n, i+(kernel/2)+1)]
        mn = np.median(seg)
        if abs(res[i] - mn) > threshold * np.std(seg):
            res[i] = mn
    return np.array(res)

#np.savetxt('gout.txt',gout)
#np.savetxt('membrane.txt',membrane)
#np.savetxt('pulseout.txt',pulseout)
#np.savetxt('lpf.txt',lpf)

    
lpf = selective_median_filter(lpf, kernel=31, threshold=2)

index_up = np.where(gout>0.05)
index_dn = np.where(gout<-0.12)
    
timemm = 1.2 # time in millisecond for the scope    
    
figure()
#subplot(3,1,1)
plot(time,membrane, label='LNA output')
#plot(time_lpf,lpf, label='Band-Pass output')
xlabel('Time [ms]')
ylabel('V')
ylim(-0.42,0.49)
xticks(np.linspace(0,time_max,7),np.linspace(0,timemm,7))
xlim([20000,40000])
#xlim([0,0.83])
#xlim([0.45,0.71])
yticks(np.linspace(-0.4,0.6,10),np.linspace(0+0.4,1.2,9))
legend(loc='best')


subplot(3,1,2)
plot(time_gout,gout, 'g-', linewidth=0.2)
plot(time_gout,gout, 'g--', label='Asynch Delta Modulator Output')
xticks(np.linspace(0,time_max,7),np.linspace(0,timemm,7))
xlim([20000,40000])
#plot(signal[:,1],signal[:,0],'o')
#xlim([0,44000])
#ylim(0.7,1.2)
#xlim([0.45,0.71])
ylabel('V')
xlabel('Time [ms]')
ylabel('V')
legend(loc='best')

subplot(3,1,3)
plot(membrane,label='LNA output')
plot(signal[:,1],signal[:,0]-np.mean(signal[:,0]),'o', label='Asynch Delta Modulator Output')
xticks(np.linspace(0,time_max,7),np.linspace(0,timemm,7))
xlim([20000,40000])
#xticks(np.linspace(0,time_max,7),np.linspace(0,timemm,7))
xlabel('Time [ms]')
ylabel('V')
legend(loc='best')

