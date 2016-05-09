''' 
2013
    author: federico corradi , federico@ini.phys.ethz.ch
	information theory analysis toolkit:
        - estimating embedding dimension of a dynamical system
        - estimating delay lag of a dynamical system
        - attractor reconstruction
        - phase space reconstruction
        - largest lyapunov exponent estimation with rosenstein algorithm
        - recurrence plot analysis
'''

import numpy as np
from time import time
import os
from pylab import *
import scipy
from scipy.stats import stats
import itertools

import zlib 
import bz2
from sets import Set

from spectrum import *

def power_spectra_mutitaper(spk_time,division=10,NW_a=2.5, k_a=2,met='adapt'):
    
    #make a sampled vector with zeros and ones
    #regular neurons
    #make a vector with zeros and ones r_spikes 
    time_max = np.max(spk_time)
    dt_small = (np.min(np.diff(spk_time))/division)
    npoints = np.ceil(time_max/dt_small)
    time_edges = np.linspace(0,time_max,int(npoints))
    r_spikes = np.zeros(npoints)

    counter_spk = 0
    for i in range(int(npoints)-1):
        #we check all time_edges and if we have a spike we put a one in r_spike vector
        if( np.logical_and(spk_time[counter_spk]>=time_edges[i],spk_time[counter_spk]<time_edges[i+1])):
            #we got a spike in this time_edge
            r_spikes[i] = 1
            counter_spk += 1

    res = pmtm(r_spikes, NW=NW_a, k=k_a, show=False, method=met)
    freqs = np.linspace(0,len(res)/10,len(res))

    #semilogy(freqs,res*np.power(10,7))

    return res,freqs

def covariance(x, k):
    N = len(x) - k
    return (x[:-k] * x[k:]).sum() / N
 
def plot_x_and_psd_with_estimated_omega(x, sample_step=1, dt=1.0):
    y = x[::sample_step]
    F = freq(x, sample_step, dt)
    T = 1.0 / F

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

def fit_exp_nonlinear(t, y):
    import scipy as sp
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=1000)
    A, K, C = opt_parms
    return A, K, C

def model_func(t, A, K, C):
    return A * np.exp(K * t) + C

def exp_fit_autocorr(signal):
    autocorr_signal = np.correlate(signal, signal, mode='full')
    norm_autocorr = autocorr_signal/max(autocorr_signal)
    norm_autocorr = norm_autocorr[norm_autocorr.size/2:]
    t = np.linspace(1,1000,len(norm_autocorr))
    a,k,c = fit_exp_nonlinear(t,np.array(norm_autocorr))
    fit_y = model_func(t, a, k, c)

    plot(norm_autocorr)
    xlabel(r'$\tau$', fontsize=20)
    ylabel(r'$\Phi(\tau)$', fontsize=20)

    plot(t, norm_autocorr, t, fit_y, (A0, K0, C0), (A, K, C0)) 

def RQA(indata, delay, embed, radius, line_min, scale_type, first, last):
    ''' perform RQA analysis
    indata: time serie  return full_rms
    delay: embedding delays
    embed: embedding dimensions
    radius: threshold for recurrency matrix 
    lineMin: minimum lenght line
    scaleType: 1 = scale matrix dimension to mean distance
               0 = scale matrix dimension to maximum distance
    first: starting vector
    last: ending vector
    ---------
	return:
	matrix
	percent_recur
	percent_determ
	percent_laminarity
	trapping_time
	ent 
	entropy_data_counts
	entropy_data_bins
    es: RQA(indata,15,5,30,3,1,1,len(indata)-delay*2)
    '''

    data = indata[first : last+((embed-1)*delay)]
    v = delay_vectors(indata,delay,embed)

    #this will need to be speeded up
    upper_triangle_dm, num_point_in_triangle = create_distance_matrix(v)

    r,c = np.shape(upper_triangle_dm)         
    x = upper_triangle_dm*2.0

    x_in1,x_in2 = np.where(np.isnan(x)) 
    x[x_in1,x_in2] = 0
    sum_d = np.sum(x)
    sum_d = np.sum(sum_d)
    mean_d = sum_d / (r * c)

    #compute distance
    upper_triangle_dm_c = upper_triangle_dm.copy()
    upper_triangle_dm_c[x_in1,x_in2] = 0
    max_d = np.max(upper_triangle_dm_c)

    scaled_upper_triangle_dm, scaled_radius = rescale_dm(upper_triangle_dm, scale_type, mean_d, max_d, radius)
   
    #apply threshold and compute recurrence matrix true false
    upper_triangle_rm = scaled_upper_triangle_dm <= scaled_radius
   
    num_recur = np.sum(upper_triangle_rm) 
   
    percent_recur = (num_recur / float(num_point_in_triangle)) * 100.0 

    line_lenghts,da_slope,trend_distance,trend_recur,trend_p = analyze_diagonals(upper_triangle_rm,0)

    line_lengths, recur_points_not_in_lines, recur_points_in_lines = process_points_in_lines(line_lenghts, line_min)    

    num_lines = len(line_lenghts)
    max_line = np.max(line_lenghts)
    percent_determ = (recur_points_in_lines/ float(num_recur)) * 100.0
    
    ent, entropy_data_counts, entropy_data_bins = compute_entropy(line_lenghts, max_line, num_lines)

    rot90_upper_triangle_rm = np.rot90(upper_triangle_rm)

    vert_line_lengths = analyze_verticals(rot90_upper_triangle_rm)

    vert_line_lengths, recur_not_in_vert_lines, recur_in_vert_lines = process_points_in_lines(vert_line_lengths, line_min)

    percent_laminarity = (recur_in_vert_lines / float(num_recur)) * 100.0

    trapping_time = np.mean(vert_line_lengths)

    full_rm = upper_triangle_rm + upper_triangle_rm.T

    full_rm = np.rot90(full_rm)

    #r,c = np.shape(full_rm)
    #binary_one = np.zeros([r,c])
    #inx, iny = np.nonzero(full_rm)
    #binary_one[inx,iny] = 1

    return full_rm,percent_recur,percent_determ,percent_laminarity,trapping_time,ent,entropy_data_counts,entropy_data_bins

def analyze_verticals(half_rm):
    '''rqa computations of vertical lines'''

    r,c = np.shape(half_rm)
    num_lines = 0
    line_lenghts = []
    for j in range(0,c):
        num_in_line = 0
        i = 0
        while ( i<r ):
            #exclude LOI
            if(half_rm[i,j] == 1):
                num_in_line += 1
                i += 1
            if(num_in_line > 0):
                if( (half_rm[i-1,j] == 1 and i-j == r) or half_rm[i,j] == 0 ):
                    num_lines += 1
                    line_lenghts.append(num_in_line)
                    i += 1
                    num_in_line = 0
            else:
                i += 1

    line_lenghts = np.array(line_lenghts)
    return line_lenghts


def compute_entropy(line_lenghts,max_line,num_lines):
    '''entropy computation for RQA '''
    num_bins = np.linspace(0,max_line,max_line)
    line_lenght_all_counts, line_lenght_all_bins = np.histogram(line_lenghts, num_bins)

    #remove all zeros
    logical_array = line_lenght_all_counts != 0
    line_lenght_counts = line_lenght_all_counts[logical_array]
    line_lenght_bins = line_lenght_all_bins[logical_array]

    dasum = 0.0
    for i in range(0,len(line_lenght_bins)):
        prob = line_lenght_counts[i] / float(num_lines)
        temp = prob * np.log2(prob)
        dasum += temp

    return -dasum, line_lenght_counts, line_lenght_bins  

def process_points_in_lines(line_lenghts_in, line_min):
    ''' recurrence plot analysis '''

    recur_points_not_in_lines = 0
    recur_points_in_lines = 0

    for i in range(len(line_lenghts_in)):
        if line_lenghts_in[i] < line_min:
            recur_points_not_in_lines += line_lenghts_in[i]
        else:
            recur_points_in_lines += line_lenghts_in[i]


    logics = line_lenghts_in > line_min
    line_lenghts_out = line_lenghts_in[logics]

    return line_lenghts_out, recur_points_not_in_lines, recur_points_in_lines

def analyze_diagonals(half_rm, line_min):
    ''' recurrence plot analysis 
    half_rm : upper triangle of a recurrence matrix
    line_min : minimum lenght that constitues a line
    --------------
    return:
    line_lenght: an array where each element is the lenght of a diagonal line
    da_slope: % local recurrent vs displacement : TREND
    '''

    r,c = np.shape(half_rm)
    num_lines = 0
    diagonal_count = 0
    line_lenghts = []

    #start from 1 so that the LOI is not included
    for count in range(1,c):
        i = 0 
        j = count
        #in diagonal
        #check for line 
        num_in_line = 0
        num_point_in_current_diagonal = 0
        while (j < c): 
            #count number of points in a diagonal
            num_point_in_current_diagonal = num_point_in_current_diagonal + 1
            if half_rm[i,j] == 1:
                num_in_line += 1
                i += 1
                j += 1
            if (j >=c ):
                break
            if num_in_line > 0:
                #if this is last column and it is at 1
                if( (half_rm[i-1,j-1] == 1 and j-i == c) or half_rm[i,j] == 0 ):
                    #line has stopped
                    line_end_flag = 1
                    #increase counter 
                    num_lines += 1
                    #store the lenght of the line
                    line_lenghts.append(num_in_line) 
                    diagonal_count += 1
                    #increase counters and move on to check next point
                    i += 1
                    j += 1
                    #reset line lenght counter
                    num_in_line = 0
            else:
                i += 1
                j += 1
    #now we compute the trend
    rdl = [] #recurrence in diagonal line = percentage of recurrence in each diagonal line (c-k)
    for k in range(1,c-1):
        tmp = np.sum(np.diag(half_rm,k))/float(c-k)*100.0
        rdl.append(tmp) 
    rdl = np.array(rdl)
    line_lenghts = np.array(line_lenghts)
    p = np.polyfit(np.linspace(2,c,c-2),rdl,1)
    da_slope = p[0]

    return line_lenghts,da_slope,np.linspace(2,c,c-2),rdl,p

def rescale_dm(upper_dm, use_mean_dist, mean_d, max_d, radius):
    ''' rescale distance matrix '''    
    if use_mean_dist == 1:
        scaled_upper_dm = np.divide(upper_dm,float(mean_d))
    else:
        scaled_upper_dm = np.divide(upper_dm,float(mean_d))

    scaled_radius = (radius/100.0)

    return scaled_upper_dm, scaled_radius

def create_distance_matrix(v):
    '''create distance matrix
    only half matrix is computed - it is symmetrix on the LOI - line of identity
    v: embedded time series matrix
    ---------
    return:
    dm: distance matrix
    num_point_in_triangle: number of points in the upper triangle
    '''    

    rows,cols = np.shape(v)   
    num_vectors = rows    

    #init distance matrix with nan
    dm = np.zeros([num_vectors,num_vectors])
    dm[:] = np.NAN 

    num_point_in_triangle = 0.0
    sum_count = 0.0
    sum_d = 0.0
    max_d = 0.0

    for j in range(num_vectors):
        for i in range(num_vectors):
            if i < j:  #only compute half
                num_point_in_triangle = num_point_in_triangle +1
                tmp_sum = 0.0
                vector_a = v[i,:]
                vector_b = v[j,:]    

                #euclidian distance is used
                tmp_2 = 0.0
                #tmp_1 = 0.0
                for k in range(cols):
                    tmp_1 = np.power((vector_a[k]-vector_b[k]),2)
                    tmp_2 = tmp_2 + tmp_1
                euclid = np.sqrt(tmp_2)
                dm[i,j] = euclid

    return dm,num_point_in_triangle

def delay_vectors(indata,delay,embed):
    ''' internal function that computes delay vectors
    used for RQA
    indata: time series
    delay: integer delay
    embed: dimensions
    ----
    return:
    v: time delay matrix
    '''

    num_points = len(indata) - (embed-1)*delay
    v = np.zeros([embed,num_points])

    for i in range(embed):
        v[i,:] = indata[i*delay:num_points+i*delay]    

    v = np.transpose(v)

    return v

def ismember(a, b):
    # tf = np.in1d(a,b) # for newer versions of numpy
    tf = np.array([i in b for i in a])
    u = np.unique(a[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
    return tf, index

def lyapunov_rosenstein(x,m,tao,meanperiod,maxiter, dt=0.01):
    ''' this function estimate the lyapunov exponent of a time series x
    x: time series
    m: embedding dimension
    tao: time delay
    meanperiod: mean period
    maxiter: max number of iterations
    dt : used to calculate sampling frequency 1./dt - time step of integration -
    ------
    return:
    lle: largest lyaponov exponent
    dd: 
    ff: polyfit 
    ref: Rosenstein M.T., Collins J.J and De Luca C.J 1993 A pratical
    method for calculating largest Lyapunov exponents from small data sets. Phys. D.
    '''
    
    nn = len(x)
    mm = nn-(m-1)*tao

    yy = psr_embedded_dim(x,m+1,tao,mm)

    neardis = np.zeros(mm)
    nearpos = np.zeros(mm)
    dd = np.zeros(maxiter)

    for i in range(mm):
        x0 = np.ones([mm,1])*yy[i,:]
        distance = sqrt(sum(pow((yy-x0),2),axis=1))
        for j in range(mm):
            if(np.abs(j-i)<=meanperiod):
                distance[j]=1e10 # far away...
        neardis[i] = min(distance)
        nearpos[i] = int(np.where(distance == min(distance))[0][0])

    for k in range(maxiter):
        maxind=mm-k
        evolve=0
        pnt=0
        for j in range(mm):
            if (j<maxind and nearpos[j]<= maxind):
                dist_k = np.sqrt(np.sum(np.power(yy[j+k-1,:]-yy[nearpos[j]+k-1,:],2)))  
                if(dist_k != 0.0):
                    evolve = evolve+log(dist_k)
                    pnt += 1
        if pnt > 0:
            dd[k] = evolve/float(pnt)
        else:
            dd[k] = 0

        
    fs = 1./dt #100.0 #sampling frequency
    tlinear = np.linspace(0,len(dd)-1,len(dd)) 
    ff = np.polyfit(tlinear,dd,1)
    polynomial = np.poly1d(ff)
    ys = polynomial(tlinear)
    #plot(tlinear,ys,'g-')
    #plot(tlinear,dd)
    #xlabel('time', fontsize=20)
    #ylabel('<ln(divergence)>', fontsize=20)
    #show()
    lle = ff[0]

    return lle,dd,ff

def fnn_embedded_dim(x,tao,mmax,rtol=15,atol=2):
    ''' this function estimate embedding dimension with the method
    of the nearest neighbours.
    ref: D.I. Abarnabel determining embedding dimension for phase-space
    reconstruction using a geometrical construction. Phys ReV. A. 1992
    x: time series
    tao: time delay - could be estimated with mutual information -
    mmax: maximum number of embedding dimension
    '''

    nn = len(x)
    ra = std(x,axis=0)

    fnn = np.zeros(mmax)
    for m in range(mmax):
        mm = int(nn-m*(tao))
        yy = psr_embedded_dim(x,m+1,tao,mm)

        for n in range(mm):
            #y0 = np.ones(mm)*yy[n,:]
            
            a1 = np.shape(yy[n,:])
            a1 = int(np.array(a1))
            y0 = np.ones([mm,a1])
            for this_dim in range(a1):
                y0[:,this_dim] = y0[:,this_dim]*yy[n,this_dim]

            distance = sqrt(sum(pow((yy-y0),2),axis=1))
            neardis= np.sort(distance)
            nearpos = np.argsort(distance)
           
            dd = np.abs(x[n+m*tao]-x[nearpos[1]+m*tao])
            rr = np.sqrt(np.power(dd,2)+np.power(neardis[1],2))
            if( dd / neardis[1] > rtol or rr/ra > atol):
                fnn[m] += 1

    fnn=(fnn/fnn[0])*100

    return fnn

def psr_embedded_dim(x,m,tao,npoint):
    ''' phase space reconstruction 
    x: time series
    m: embedding dimension
    tao: time delay
    npoint: total number of points
    return y: M x mm matrix
    '''
    n = len(x)
    mm = npoint #or m=n-(m-1)*tao
    
    y = np.zeros([mm,m])

    for i in range(m):
        idx = np.linspace(0,mm-1,mm)
        tmp_idx = idx+((i-1)*tao)
        b = tmp_idx.astype(int)
        y[:,i] = x[b].T

    return y

def mutual_information(signal,max_tau,partitions=16):
    ''' this function compute the mutual information of 
    a given time series signal for a number of delays tau=0,..,maxtau
    The probabilities are evaluated partitioning the domain and the number of partition
    in one dimension is given by partitions
    the output is a two coulmn numpy array: first column has delays, second column has
    the mutual information. '''
    
    n = len(signal)
    h1v = np.zeros(partitions) # p(x(t+tau))
    h2v = np.zeros(partitions) # p(x(t))
    h12m = np.zeros([partitions,partitions]) # p(x(t+tau),x(t))

    #normalize data
    xmin = np.min(signal)
    xmax = np.max(signal)
    imax =  np.where(signal == xmax) #imax indexes
    signal[imax[0][0]] = xmax + (xmax-xmin)*10/100000000000.0 # to avoid multiple exact minima

    yV = (signal-xmin)/(xmax-xmin) 
    
    array_v = np.floor(yV*partitions)+1 #array of partitions 1,..,partitions
    array_v[imax] = partitions # set the max in the last partition

    mut_m = np.zeros([max_tau+1,2])
    mut_m[0:max_tau+1,0] = np.linspace(0,max_tau,max_tau+1).T

    for tau in range(max_tau):
        ntotal = n-tau
        mut_s = 0.0
        for i in range(partitions):
            for j in range(partitions):
                h12m[i,j] = len(np.logical_and(array_v[tau:n] == i+1, array_v[0:n-tau] == j+1).nonzero()[0]) 
                #index_for = np.where(array_v[tau:n] == i+1)
                #index_back= np.where(array_v[0:n-tau] == j+1)
                #h12m[i,j]=len(ismember(index_for[0],index_back[0]))
                #len( np.where(np.logical_and(  array_v[tau:n] == i+1, array_v[0:n-tau] == j+1 ) ))
        
        for i in range(partitions):
            h1v[i] = np.sum(h12m[:,i])
            h2v[i] = np.sum(h12m[i,:])

        for i in range(partitions):
            for j in range(partitions):
                if h12m[i,j] > 0:
                    mut_s = mut_s+(h12m[i,j]/ntotal)*log(h12m[i,j]*ntotal/(h1v[i]*h2v[j]))

        mut_m[tau,1] = mut_s

    return mut_m

def entropy(X):
    #calculate entropy of an array
    probs = [np.mean(X == c) for c in set(X)]
    return np.sum(-p * np.log2(p) for p in probs)

def joint_entropy(*X):
    #calculate joint entropy
    return np.sum(-p * np.log2(p) if p > 0 else 0 for p in
              (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
                for classes in itertools.product(*[set(x) for x in X])))

def shannon(st):
    #get a string and compute its shannon entropy

    stList = list(st)
    alphabet = list(Set(stList)) # list of symbols in the string
    print 'Alphabet of symbols in the string:'
    print alphabet
    print
    # calculate the frequency of each symbol in the string
    freqList = []
    for symbol in alphabet:
        ctr = 0
        for sym in stList:
            if sym == symbol:
                ctr += 1
        freqList.append(float(ctr) / len(stList))
    print 'Frequencies of alphabet symbols:'
    print freqList
    print
    # Shannon entropy
    ent = 0.0
    for freq in freqList:
        ent = ent + freq * math.log(freq, 2)
    ent = -ent
    print 'Shannon entropy:'
    print ent
    print 'Minimum number of bits required to encode each symbol:'
    print int(math.ceil(ent))

    return ent

def kolmogorov(s):
    l = float(len(s))
    compr = zlib.compress(s)
    c = float(len(compr))
    return c/l 
#    return float(len(bz2.compress(s)))/float(len(s))

def simple_frequency_spectrum(x):
    """
    Very simple calculation of frequency spectrum with no detrending,
    windowing, etc. Just the first half (positive frequency components) of
    abs(fft(x))
    """
    spec = np.absolute(np.fft.fft(x))
    spec = spec[:len(x)/2] # take positive frequency components
    spec /= len(x)         # normalize
    spec *= 2.0            # to get amplitudes of sine components, need to multiply by 2
    spec[0] /= 2.0         # except for the dc component
    return spec


def ccf(x, y, axis=None):
    """
	from neurotools

    Computes the cross-correlation function of two series x and y.
    Note that the computations are performed on anomalies (deviations from
    average).
    Returns the values of the cross-correlation at different lags.
        
    Inputs:
        x    - 1D MaskedArray of a Time series.
        y    - 1D MaskedArray of a Time series.
        axis - integer *[None]* Axis along which to compute (0 for rows, 1 for cols).
               If `None`, the array is flattened first.
    
    Examples:
        >> z= arange(1000)
        >> ccf(z,z)

    """
    assert x.ndim == y.ndim, "Inconsistent shape !"
#    assert(x.shape == y.shape, "Inconsistent shape !")
    if axis is None:
        if x.ndim > 1:
            x = x.ravel()
            y = y.ravel()
        npad = x.size + y.size
        xanom = (x - x.mean(axis=None))
        yanom = (y - y.mean(axis=None))
        Fx = np.fft.fft(xanom, npad, )
        Fy = np.fft.fft(yanom, npad, )
        iFxy = np.fft.ifft(Fx.conj()*Fy).real
        varxy = np.sqrt(np.inner(xanom,xanom) * np.inner(yanom,yanom))
    else:
        npad = x.shape[axis] + y.shape[axis]
        if axis == 1:
            if x.shape[0] != y.shape[0]:
                raise ValueError, "Arrays should have the same length!"
            xanom = (x - x.mean(axis=1)[:,None])
            yanom = (y - y.mean(axis=1)[:,None])
            varxy = np.sqrt((xanom*xanom).sum(1) * (yanom*yanom).sum(1))[:,None]
        else:
            if x.shape[1] != y.shape[1]:
                raise ValueError, "Arrays should have the same width!"
            xanom = (x - x.mean(axis=0))
            yanom = (y - y.mean(axis=0))
            varxy = np.sqrt((xanom*xanom).sum(0) * (yanom*yanom).sum(0))
        Fx = np.fft.fft(xanom, npad, axis=axis)
        Fy = np.fft.fft(yanom, npad, axis=axis)
        iFxy = np.fft.ifft(Fx.conj()*Fy,n=npad,axis=axis).real
    # We juste turn the lags into correct positions:
    iFxy = np.concatenate((iFxy[len(iFxy)/2:len(iFxy)],iFxy[0:len(iFxy)/2]))
    return iFxy/varxy

def source_entropy_rate(binary_seq,words_len=10,bins=10):
    n_words = len(binary_seq)/words_len
    norm_count_pi = np.zeros(words_len)
    for l in range(n_words):
        for i in range(words_len):
            norm_count_pi[i] += binary_seq[(l*words_len):(l+1)*words_len][i]	
    norm_term = max(norm_count_pi)
    norm_count_pi = norm_count_pi/norm_term

    entropy_tmp = 0.0
    for i in range(words_len):
        entropy_tmp += norm_count_pi[i]*np.log2(norm_count_pi[i])  
    entropy = -1.0/(words_len*bins)*entropy_tmp

    return entropy

def create_analog_from_binary(binary_state,length=5):
    
    new_length = len(binary_state)/length
    new_analog_array = np.zeros(new_length) 

    for i in range(new_length):
        for n in range(length):
            if binary_state[(i*length):(i+1)*length][n] == 1:
                new_analog_array[i] += pow(2,n)
            elif binary_state[(i*length):(i+1)*length][n] == -1:
                new_analog_array[i] += -pow(2,n)

    return new_analog_array

def check_distrib_double(d, length_seq = 8,  bins=10, thr_att=10, thr_low=8):
    
    figure_counter = int(time()) 
    figure_directory = d + 'figures/'
    if os.path.isdir(figure_directory):
        print 'figure directory already exists..'
    else:
        os.mkdir(figure_directory)

    #raw_data = np.loadtxt(d+'out_mon_att.txt')#out_mon_att
    out_mon_att = np.load(d+'out_mon_att.pickle')
    out_mon_att_1 = np.load(d+'out_mon_att_1.pickle')
    raw_data = out_mon_att.sl.raw_data()
    mean_rate = out_mon_att.sl.mean_rates()
    t_start = np.min(raw_data[:,0])
    t_stop = np.max(raw_data[:,0])

    out_mon = np.load(d+'out_mon.pickle')
    out_mon_neu_z = np.load(d+'out_mon_neu_z.pickle')

    raw_data = out_mon_att.sl.raw_data()
    mean_rate = out_mon_att.sl.mean_rates()
    t_start = np.min(raw_data[:,0])
    t_stop = np.max(raw_data[:,0])

    firing_rates = out_mon_att.firing_rates(time_bin=bins)
    index_att = np.nonzero(firing_rates>thr_att)
    attractor_indexes = np.array_split(index_att[0],np.where(np.diff(index_att[0])!=1)[0]+1)#find continuous element of the array
    raw_data_att = out_mon_att.sl.raw_data()  #get raw data

    firing_rates_low = out_mon.firing_rates(time_bin=bins)
    index_low = np.nonzero(firing_rates_low<thr_low)
    low_indexes = np.array_split(index_low[0],np.where(np.diff(index_low[0])!=1)[0]+1)#find continuous element of the array
    raw_data_low = out_mon.sl.raw_data()  #get raw data

    firing_rates_1 = out_mon_att_1.firing_rates(time_bin=bins)
    index_att_1 = np.nonzero(firing_rates_1>thr_att)
    attractor_indexes_1 = np.array_split(index_att_1[0],np.where(np.diff(index_att_1[0])!=1)[0]+1)#find continuous element of the array
    raw_data_att_1 = out_mon_att_1.sl.raw_data()  #get raw data


    #find lenght of low
    n_low = np.shape(low_indexes)
    n_low = np.array(n_low)
    n_low = n_low[0]
    index_low_array = []
    for i in range(n_low):
            index_low_array.extend(low_indexes[i])
    index_low_array = np.array(index_low_array)

    #find index high
    n_hi = np.shape(attractor_indexes)
    n_hi = np.array(n_hi)
    n_hi = n_hi[0]
    index_hi_array = []
    for i in range(n_hi):
        index_hi_array.extend(attractor_indexes[i])
    index_hi_array = np.array(index_hi_array)

    #find index high
    n_hi_1 = np.shape(attractor_indexes_1)
    n_hi_1 = np.array(n_hi_1)
    n_hi_1 = n_hi_1[0]
    index_hi_1_array = []
    for i in range(n_hi_1):
        index_hi_1_array.extend(attractor_indexes_1[i])
    index_hi_1_array = np.array(index_hi_1_array)


    #from hi and low create analog values
    max_hi = np.max(index_hi_array)
    max_hi_1 = np.max(index_hi_1_array)
    max_low = np.max(index_low_array)
    max_tot = np.max([max_hi, max_low])

    binary_state = np.zeros(max_tot)
    binary_state[index_hi_array[:-1:]] = 1
    binary_state[index_hi_1_array[:-1:]] = -1

    analog_array = create_analog_from_binary(binary_state,length=length_seq)

    return analog_array,binary_state


def check_distrib_single(d, length_seq = 8,  bins=10, thr_att=10, thr_low=8):
    
    figure_counter = int(time()) 
    figure_directory = d + 'figures/'
    if os.path.isdir(figure_directory):
        print 'figure directory already exists..'
    else:
        os.mkdir(figure_directory)

    #raw_data = np.loadtxt(d+'out_mon_att.txt')#out_mon_att
    out_mon_att = np.load(d+'out_mon_att.pickle')
    raw_data = out_mon_att.sl.raw_data()
    mean_rate = out_mon_att.sl.mean_rates()
    t_start = np.min(raw_data[:,0])
    t_stop = np.max(raw_data[:,0])

    out_mon = np.load(d+'out_mon.pickle')
    out_mon_neu_z = np.load(d+'out_mon_neu_z.pickle')

    raw_data = out_mon_att.sl.raw_data()
    mean_rate = out_mon_att.sl.mean_rates()
    t_start = np.min(raw_data[:,0])
    t_stop = np.max(raw_data[:,0])

    firing_rates = out_mon_att.firing_rates(time_bin=bins)
    index_att = np.nonzero(firing_rates>thr_att)
    attractor_indexes = np.array_split(index_att[0],np.where(np.diff(index_att[0])!=1)[0]+1)#find continuous element of the array
    raw_data_att = out_mon_att.sl.raw_data()  #get raw data

    firing_rates_low = out_mon.firing_rates(time_bin=bins)
    index_low = np.nonzero(firing_rates_low<thr_low)
    low_indexes = np.array_split(index_low[0],np.where(np.diff(index_low[0])!=1)[0]+1)#find continuous element of the array
    raw_data_low = out_mon.sl.raw_data()  #get raw data

    #find lenght of low
    n_low = np.shape(low_indexes)
    n_low = np.array(n_low)
    n_low = n_low[0]
    index_low_array = []
    for i in range(n_low):
            index_low_array.extend(low_indexes[i])
    index_low_array = np.array(index_low_array)

    #find index high
    n_hi = np.shape(attractor_indexes)
    n_hi = np.array(n_hi)
    n_hi = n_hi[0]
    index_hi_array = []
    for i in range(n_hi):
            index_hi_array.extend(attractor_indexes[i])
    index_hi_array = np.array(index_hi_array)


    #from hi and low create analog values
    max_hi = np.max(index_hi_array)
    max_low = np.max(index_low_array)
    max_tot = np.max([max_hi, max_low])

    binary_state = np.zeros(max_tot)
    binary_state[index_hi_array[:-1:]] = 1

    analog_array = create_analog_from_binary(binary_state,length=length_seq)

    return analog_array,binary_state


def fano_factor_isi(isi):
    """ 
    Return the fano factor of this spike trains ISI.

    The Fano Factor is defined as the variance of the isi divided by the mean of the isi

    See also
        isi, cv_isi

    from neurotools
    """
    if len(isi) > 0:
        fano = numpy.var(isi)/numpy.mean(isi)
        return fano
    else: 
        raise Exception("No spikes in the SpikeTrain !")
