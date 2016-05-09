import matplotlib
from pylab import *
from scipy import signal
import math
import wave

def open_wavefile(wavefile):
    spf = wave.open(wavefile,'r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    return signal


    
# some constants
spf = wave.open(wavefile,'r')
framerate = spf.getframerate() #samples / seconds
frames = spf.getnframes() 
rate = spf.getframerate()
duration = frames / float(rate)
samp_rate = frames / float(duration)
sim_time = duration
nsamps = int(samp_rate*sim_time)
cuttoff_freq = 50
t = np.linspace(0, sim_time, nsamps)

#################
### WAVEFILE
#################
wavefile = '/home/federico/projects/work/trunk/data/Insects/Insect Neurophys Data/Hackenelektrodenableitung_Grillen2_filter.wav'
rec = open_wavefile(wavefile)



sz = 44100 # Read and process 1 second at a time.
da = np.fromstring(spf.readframes(44100), dtype=np.int16)
left, right = da[0::2], da[1::2]
lf, rf = np.fft.rfft(left), np.fft.rfft(right)

plt.figure(1)
a = plt.subplot(211)
r = 2**16/2
a.set_ylim([-r, r])
a.set_xlabel('time [s]')
a.set_ylabel('sample value [-]')
plt.plot(np.linspace(0,sim_time,len(left)), left)
b = plt.subplot(212)
b.set_xscale('log')
b.set_xlabel('frequency [Hz]')
b.set_ylabel('|amplitude|')
plt.plot(abs(lf))


lowpass = 21 # Remove lower frequencies.
highpass = 9000 # Remove higher frequencies.

lf[:lowpass], rf[:lowpass] = 0, 0 # low pass filter (1)
lf[55:66], rf[55:66] = 0, 0 # line noise filter (2)
lf[highpass:], rf[highpass:] = 0,0 # high pass filter (3)
nl, nr = np.fft.irfft(lf), np.fft.irfft(rf) # (4)
ns = np.column_stack((nl,nr)).ravel().astype(np.int16)

###################
### SEPARATE AUDIO
###################
fig = figure()

xfreq = np.fft.fft(rec)
fft_freqs = np.fft.fftfreq(nsamps, d=1./samp_rate)
fig.add_subplot(233)
loglog(fft_freqs[0:nsamps/2], np.abs(xfreq)[0:nsamps/2])
title('Filtered - Frequency Domain')
grid(True)


# design filter
norm_pass = cuttoff_freq/(samp_rate/2)
norm_stop = 1.5*norm_pass
(N, Wn) = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30, analog=0)
(b, a) = signal.butter(N, Wn, btype='low', analog=0, output='ba')
print("b="+str(b)+", a="+str(a))

# filter frequency response
(w, h) = signal.freqz(b, a)
fig.add_subplot(131)
loglog(w, np.abs(h))
title('Filterquency Response')
text(2e-3, 1e-5, str(N)+"-th order Butterworth filter")
grid(True)

# filtered output
#zi = signal.lfiltic(b, a, x[0:5], x[0:5])
#(y, zi) = signal.lfilter(b, a, x, zi=zi)
y = signal.lfilter(b, a, rec)
fig.add_subplot(235)
plot(t, y)
title('Filterput - Time Domain')
grid(True)

# output spectrum
yfreq = np.fft.fft(y)
fig.add_subplot(236)
loglog(fft_freqs[0:nsamps/2], np.abs(yfreq)[0:nsamps/2])
title('Filterput - Frequency Domain')
grid(True)

show()
