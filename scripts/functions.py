import sys
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

def find_closest(A, target):
	#A must be sorted
	idx = A.searchsorted(target)
	idx = np.clip(idx, 1, len(A)-1)
	left = A[idx-1]
	right = A[idx]
	idx -= target - left < right - target
	return idx

if __name__=="__main__":
    series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
    print peakdet(series,1)

##### Mean firing rate of each neuron
def meanNeuFiring(SpikeTrain, n_neurons,simulation_time):
	ArraySpike = np.array(SpikeTrain)
	MeanRate   = np.zeros([len(n_neurons)])
	for i in range(len(n_neurons)):
		Index_n = np.where(np.logical_and(ArraySpike[:,1] == n_neurons[i], np.logical_and(ArraySpike[:,0] > simulation_time[0] , ArraySpike[:,0] < simulation_time[1] )) )
		MeanRate[i] = len(Index_n[0])*1000.0/(simulation_time[1]-simulation_time[0]) # time unit: ms
	return MeanRate




