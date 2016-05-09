import wave
import sys
spf = wave.open('/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/Hackenelektrodenableitung_mecopoda_elongata_chirper_2.wav','r')
#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
time_orig=np.linspace(0, len(signal)/fs, num=len(signal))


figure()
plot(time_orig,signal)
