import numpy as np
from pylab import *
import matplotlib 
import pyNCS
import pyNCS.AerViewer
from pyNCS.neurosetup import NeuroSetup
import sys
# here you need to confiture your mapper
sys.path.append('mapper/')
sys.path.append('_utils/')
sys.path.append('_utils/mapperlib')
sys.path.append('_utils/biasgenlib')

import mapper
import AERmn256r1

#init mappings
offsetfirstchip = 500
offset_usb = 2000
offset_adcs = 2600
offset_adcs1 = 3100
offset_adcs2 = 3300
print "Init mapper ..."
mapper.init_mappings(offsetfirstchip,offset_usb,offset_adcs, offset_adcs1, offset_adcs2)


####PROGRAM SYNAPTIC MATRIC
matrix_programmable_w = np.zeros([256,256])+1
print "Program connections weights ..."
AERmn256r1.load_weight_matrix_programmable(matrix_programmable_w )
print "Program connections exc/inh ..."
matrix_programmable_exc = np.zeros([256,256])+1
AERmn256r1.load_matrix_exc_inh(matrix_programmable_exc)
matrix_learning_rec = np.zeros([256,256])
print "Program recurrent plastic connections ..."
AERmn256r1.load_connections_matrix_plastic(matrix_learning_rec)
print "Program connections ..."
matrix_connections_rec = np.ones([256,256])
AERmn256r1.load_connections_matrix_programmable(matrix_connections_rec)
print "Program plastic weights ..."
AERmn256r1.load_weight_matrix_plastic(matrix_learning_rec)


################################
# pyNCS populations
################################
import pyNCS
from pyNCS.neurosetup import NeuroSetup

neurons = np.linspace(0,255,256)

# Define file paths
prefix='./chipfiles/'
setuptype = 'setupfiles/mc_final_mn256r1_adcs.xml'
setupfile = 'setupfiles/final_mn256r1_adcs.xml'

#Create Chip objects
setup = pyNCS.NeuroSetup(setuptype,setupfile,prefix=prefix)
chip = setup.chips['mn256r1']

chip_pop = pyNCS.Population('neurons', 'target ')
chip_pop.populate_by_number(setup,'mn256r1','excitatory',256)

pulse_neu = pyNCS.Population('neurons', 'monitoring ')
pulse_neu.populate_by_id(setup,'mn256r1','adcs',[310])

up_neu = pyNCS.Population('neurons', 'monitoring ')
up_neu.populate_by_id(setup,'mn256r1','adcs',[305])

dn_neu = pyNCS.Population('neurons', 'monitoring ')
dn_neu.populate_by_id(setup,'mn256r1','adcs',[300])

#ad_mon = pyNCS.Population('neurons', 'monitoring ')
#ad_mon.populate_by_number(setup,'mn256r1','adcs',(2**16-257))

#monitors
#ad_pop = pyNCS.monitors.SpikeMonitor(ad_mon.soma)
mon_chip =  pyNCS.monitors.SpikeMonitor(chip_pop.soma)
pulse_neu_mon = pyNCS.monitors.SpikeMonitor(pulse_neu.soma)
up_neu_mon = pyNCS.monitors.SpikeMonitor(up_neu.soma)
dn_neu_mon = pyNCS.monitors.SpikeMonitor(dn_neu.soma)
setup.monitors.import_monitors([up_neu_mon,dn_neu_mon,mon_chip,pulse_neu_mon])

setup.stimulate({}, send_reset_event=False, duration=1000)




##plot ad events 
#raw_data = ad_pop.sl.raw_data()
#indexed = np.bitwise_and(raw_data[:,1].astype(int),1)
#index_up = np.where(indexed == 0)[0]
#index_dn = np.where(indexed == 1)[0]

#scatter(raw_data[index_dn,0], raw_data[index_dn,1])
#scatter(raw_data[index_up,0], raw_data[index_up,1])
#show()


def get_destination_address(neus,syn):


    syn_index = chip_pop.synapses['programmable']['syntype'].addr == syn
    neu_index = chip_pop.synapses['programmable']['neu'].addr == neus
    this_syn_neu = (syn_index &   neu_index )
    dest = chip_pop.synapses['programmable'][this_syn_neu].paddr
    n_neu = len(dest)
    #dest = (dest & 65535)

    
    return dest

##map the output to synapses of one neuron
addr_ex_neu_10 = get_destination_address(10,258)
addr_ex_neu_100 = get_destination_address(10,257)
addr_in_neu_10 = get_destination_address(11,257)
addr_in_neu_100 = get_destination_address(11,257)

post_100 = [65636,65636,65636,65636]
post_120 = [65656,65656,65656,65656]


memoffset = offset_adcs1+128
memoffset_1 = offset_adcs+128
memoffset_2 = offset_adcs2+128
pre = 1
#memoffset = mapper.program_multicast(pre,post_exc,memoffset,offset_adcs,(1<<4))
#mapper.enable_detail_mapping(2**4)
memoffset_2 = mapper.program_multicast(0,post_100,memoffset_2,offset_adcs2,(1<<5))
memoffset = mapper.program_multicast(0,post_120,memoffset,offset_adcs1,(1<<3))
memoffset_1 = mapper.program_multicast(0,post_120,memoffset_1,offset_adcs,(1<<4))
mapper.enable_detail_mapping(2**3+2**4+2**5)

up = up_neu_mon.sl.raw_data()[:,0]
down = dn_neu_mon.sl.raw_data()[:,0]
delta_dn = 0.1
delta_up = 0.1
total = np.zeros([len(up)+len(down)+1,2])
mapper.enable_detail_mapping(2**4)
countup = 0
countdown = 0
for i in range(1,len(up)+len(down)):
	if(countup != len(up) and countdown != len(down)):
		if(up[countup] < down[countdown]):
			total[i,0] = total[i-1,0]+delta_up
			total[i,1] = up[countup]
			countup = countup+1
		elif(down[countdown] < up[countup]):
			total[i,0] = total[i-1,0]-delta_dn
			total[i,1] = down[countdown]
			countdown = countdown+1


#plot(total[:,1],total[:,0]-np.mean(total[:,0]),'o')

	
# generate a perfect data set (my real data have tiny error)
def mysine(x, a1, a2, a3):
    return a1 * np.sin(a2 * x + a3)
    
xReal = total[100:1500,1] - np.min(total[100:1500,1])
yReal = ( total[100:1500,0]-np.mean(total[100:1500,0])  )  / np.max(total[100:1500,0])
    
import numpy as np
import scipy.optimize as optimize
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
pi = np.pi

yhat = fftpack.rfft(yReal)
idx = (yhat**2).argmax()
freqs = fftpack.rfftfreq(len(xReal), d = np.mean(np.diff(xReal))/(2*pi))
frequency = freqs[idx]	
	
amplitude = yReal.max()
guess = [amplitude, frequency, 0.]
print(guess)
(amplitude, frequency, phase), pcov = optimize.curve_fit(mysine, xReal, yReal, guess)

period = 2*pi/frequency
print(amplitude, frequency, phase)

xx = xReal
yy = mysine(xx, amplitude, frequency, phase)
# plot the real data
plt.plot(xReal, yReal, 'r', label = 'Real Values')
plt.plot(xx, yy , label = 'fit')
plt.legend(shadow = True, fancybox = True)
plt.show()

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


print(snr(yReal, yy))

#SNR  23.90u REFR Period
#SNR  2 kHz  9.04, 10.12, 11.15, 12.05, 9.51
#SNR  1 kHz  14.20, 17.01,  17.52, 15.57,  15.65 
#SNR  500Hz 18.09, 19.70, 20.81, 18.97, 27.52  
#SNR  200Hz 22.10, 23.38, 22.79, 21.23, 23.94
#SNR  100Hz 23.16, 21.18, 27.54, 23.7, 29.75, 26.79
#SNR  10Hz  26.24 , 21.15, 22.93 , 30.53, 21.6
ref_3 = []
#freq mean variance
ref_3.append([2000, 10.37, 1.2])
ref_3.append([1000, 15.99, 1.6])
ref_3.append([500, 21.01, 1.5])
ref_3.append([200, 22.69, 1.3])
ref_3.append([100, 25.35, 1.1])
ref_3.append([10, 24.49, 1.3])
ref_3 = np.array(ref_3)

figure()
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
errorbar(ref_3[:,0], ref_3[:,1], yerr=ref_3[:,2], marker='o',  label='Refr. 20 uA')
xscale('log')
#SNR  3.37u REFR Period
#SNR  2 kHz 9.2 , 8.82 , 9.60, 9.85, 9.53
#SNR  1 kHz 17.47, 16.55, 16.3, 14.06, 16.15
#SNR  500Hz 27, 22.44, 20.51, 23.78, 23.31
#SNR  200Hz 17.7, 19.04, 17.71, 20.08, 19.54
#SNR  100Hz 19.96, 18.35, 18.74, 17.55, 19.75  dB
#SNR  70Hz  22.44 , 19.25, 20.37, 18.57, 
#SNR  50Hz  21.04, 18.45, 17.25, 18.52,17.65   dB
#SNR  20Hz  16.7,17.01,16.9, 22.33 dB 
#SNR  10Hz  19.61, 21.64, 20.10,  19.77 , 22.98
ref_2 = []
#freq mean variance
ref_2.append([2000, 9.4, 1.8])
ref_2.append([1000, 16.1, 1.6])
ref_2.append([500, 21.4, 1.5])##
ref_2.append([200, 20.8, 1.5])
ref_2.append([100, 20.9, 1.3])
ref_2.append([70, 20.2, 1.6])
ref_2.append([50, 20.5, 1.3])
ref_2.append([20, 20.4, 1.2])
ref_2.append([10, 20.8, 1.5])
ref_2 = np.array(ref_2)
errorbar(ref_2[:,0], ref_2[:,1], yerr=ref_2[:,2], marker='d', label='Refr. 3.0 nA')
#SNR  0.2n REFR Period -> 
#SNR 2 kHz   7.51, 7.8, 8.51, 6.79
#SNR 1 kHz   9.57, 8.51 , 10.75, 10.41
#SNR 500 Hz  19.15, 16.15, 19.32, 20.42
#SNR 200 Hz  20.15, 21.15, 18.32, 26.42
#SNR 100 HZ  24, 27, 24, 23.8, 26, 23.71
#SNR 50Hz    20.35, 20.53, 18.99, 19.41 , 19.72, 20
#SNR 20Hz    13.9, 14.43, 15.93, 14.4, 15.31
#SNR 10 Hz   11.56, 12.31, 10.7, 11.5
ref_1 = []
#freq mean variance
ref_1.append([2000, 7.9, 2.1])
ref_1.append([1000, 11.0, 1.5])
ref_1.append([500, 19.0, 1.6])##
ref_1.append([200, 20.1, 1.3])
ref_1.append([100, 19.0, 1.1])
ref_1.append([70, 19.2, 1.2])
ref_1.append([50, 20.1, 1.3])
ref_1.append([20, 18.5, 1.6])
ref_1.append([10, 18.6, 1.6])
ref_1 = np.array(ref_1)
errorbar(ref_1[:,0], ref_1[:,1], yerr=ref_1[:,2], marker='<', label='Refr. 200 pA')
legend()
xlabel('Frequency [Hz]', fontsize=18)
ylabel('SNER [dB]', fontsize=18)
grid(True,which="both",ls="-", color='0.45', alpha=0.5)

