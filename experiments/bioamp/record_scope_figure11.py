   
import pyAgilent
import matplotlib
from pylab import *
import numpy as np

from numpy import NaN, Inf, arange, isscalar, asarray
import numpy as np

def peakdet(v, delta, x = None):
    """
Converted from MATLAB script at http://billauer.co.il/peakdet.html
Currently returns two lists of tuples, but maybe arrays would be better
function [maxtab, mintab]=peakdet(v, delta, x)
%PEAKDET Detect peaks in a vector
% [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
% maxima and minima ("peaks") in the vector V.
% MAXTAB and MINTAB consists of two columns. Column 1
% contains indices in V, and column 2 the found values.
%
% With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
% in MAXTAB and MINTAB are replaced with the corresponding
% X-values.
%
% A point is considered a maximum peak if it has the maximal
% value, and was preceded (to the left) by a value lower by
% DELTA.
% Eli Billauer, 3.4.05 (Explicitly not copyrighted).
% This function is released to the public domain; Any use is allowed.
"""
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return maxtab, mintab
    
osc_a = pyAgilent.Agilent(host="172.19.10.159");
osc_a._send_command('WAV:FORM RAW');

lna = osc_a._read_data_from_channel(4)
gout = osc_a._read_data_from_channel(3)
up = osc_a._read_data_from_channel(2)
dn = osc_a._read_data_from_channel(1)

time_lna = np.linspace(0,2,len(lna))
time_gout = np.linspace(0,2,len(gout))
time_up = np.linspace(0,2,len(up))
time_dn = np.linspace(0,2,len(dn))


np.savetxt('lna_time.txt', time_lna)
np.savetxt('lna.txt', lna)
np.savetxt('gout_time.txt', time_gout)
np.savetxt('gout.txt', gout)
np.savetxt('up_time.txt', time_up)
np.savetxt('up.txt', up)
np.savetxt('dn_time.txt', time_dn)
np.savetxt('dn.txt', dn)

up = np.loadtxt('up.txt')
time_up = np.loadtxt('up_time.txt')
dn = np.loadtxt('dn.txt')
time_dn = np.loadtxt('dn_time.txt')
lna = np.loadtxt('lna.txt')
time_lna = np.loadtxt('lna_time.txt')
gout = np.loadtxt('gout.txt')
time_gout = np.loadtxt('gout_time.txt')

subplot(3,1,1)
plot(time_lna,lna-np.mean(lna), label='lna output')
ylabel('Amp [V]')
subplot(3,1,2)
plot(time_gout,gout, label='gout')
ylabel('Amp [V]')
subplot(3,1,3)
#plot(time_up,up, label='up')
#plot(time_dn,dn, label='up')
xlim([0,1000])
ylim([0,1])
xlabel('Time [ms]')





