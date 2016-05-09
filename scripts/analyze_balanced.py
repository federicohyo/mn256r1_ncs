import pyNCS
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


def plot_double_chaotic_itinerancy(d):
    ''' plot chaotic itinerancy as tsuda likes '''
    figure_counter = int(time()) 
    figure_directory = d + 'figures/'
    if os.path.isdir(figure_directory):
        print 'figure directory already exists..'
    else:
        os.mkdir(figure_directory)

    #raw_data = np.loadtxt(d+'out_mon_att.txt')#out_mon_att
    out_mon_att = np.load(d+'out_mon_att.pickle')
    out_mon_att_1 = np.load(d+'out_mon_att_1.pickle')    

    a = out_mon_att.firing_rates()
    b = out_mon_att_1.firing_rates()
    figure()    
    plot(a,b)
    xlabel(r'$\nu_{1}$', fontsize=20)
    ylabel(r'$\nu_{2}$', fontsize=20)
    filename = str(figure_directory)+str(figure_counter)+str('_att_1vs_att.png')
    savefig(filename)     
    figure_counter += 1

    figure()
    H, xedges, yedges = np.histogram2d(a, b, bins=(128,128))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    imshow(H, extent=extent)
    colorbar()
    xlabel('x')
    ylabel('y')
    filename = str(figure_directory)+str(figure_counter)+str('_2dhist.png')
    savefig(filename)     
    figure_counter += 1

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
    tlinear = np.linspace(10,60,50) 
    ff = np.polyfit(tlinear,dd[10:60],1)
    polynomial = np.poly1d(ff)
    results = {}
    results['polynomial'] = polynomial.coeffs.tolist()
    correlation = np.corrcoef(tlinear,dd[10:60])[0,1]
    results['correlation'] = correlation
    results['determination'] = correlation**2
    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(tlinear,dd[10:60])
    ys = polynomial(tlinear)
    figure()
    plot(tlinear,ys,'g-')
    plot(tlinear,dd[10:60], 'o', color='black')
    xlabel(r'$Time$', fontsize=20)
    ylabel(r'$<ln(divergence)>$', fontsize=20)
    text(22, 5.256,r'$f(x) = $'+str(round(ff[0],4))+'$x+$'+str(round(ff[1],4)),horizontalalignment='center',verticalalignment='center')
    text(22, 5.24,r'$R^{2}=$'+str(round(results['determination'],4)),horizontalalignment='center',verticalalignment='center')
    show()
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
    n_words = len(binary_seq)/10
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

def study_single_neuron(d):

    figure_counter = int(time()) 
    figure_directory = d + 'figures/'
    if os.path.isdir(figure_directory):
        print 'figure directory already exists..'
    else:
        os.mkdir(figure_directory)

    out_mon_neu_z = np.load(d+'out_mon_neu_z.pickle')  
    raw_data = out_mon_neu_z.sl.raw_data()

    return raw_data[:,0]      

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

def plot_deterministic_double(d, bins=60, thr_att=10, thr_low=5):

    figure_counter = int(time()) 
    figure_directory = d + 'figures/'
    if os.path.isdir(figure_directory):
        print 'figure directory already exists..'
    else:
        os.mkdir(figure_directory)

    #plot the data
    spiketrain = pyNCS.pyST.loadtxt(d+'spiketrain_input.txt', 'spike train exp')
    spiketrain = {10:spiketrain}

    spiketrain[10].raster_plot()
    title('driver input')
    filename = str(figure_directory)+str(figure_counter)+str('_driver_input.png')
    savefig(filename)     
    figure_counter += 1
    

    #raw_data = np.loadtxt(d+'out_mon_att.txt')#out_mon_att
    out_mon_att = np.load(d+'out_mon_att.pickle')
    out_mon_att_1 = np.load(d+'out_mon_att_1.pickle')
    raw_data = out_mon_att.sl.raw_data()
    mean_rate = out_mon_att.sl.mean_rates()
    t_start = np.min(raw_data[:,0])
    t_stop = np.max(raw_data[:,0])

    out_mon = np.load(d+'out_mon.pickle')
    out_mon_neu_z = np.load(d+'out_mon_neu_z.pickle')
    out_mon_neu_z1 = np.load(d+'out_mon_neu_z1.pickle')

    #spiketrain
    raw_data = out_mon_att.sl.raw_data()
    mean_rate = out_mon_att.sl.mean_rates()
    t_start = np.min(raw_data[:,0])
    t_stop = np.max(raw_data[:,0])

    #chip activity
    out_mon_att.plot_args = {'linestyle':'--'}
    pyNCS.monitors.MeanRatePlot([out_mon_att,out_mon_att_1],time_bin=180)
    legend(["pop A","pop B"], loc=1)
    xlim(t_start-500,t_start+7500)
    ylim(0,np.max(mean_rate)+45)
    xlabel('Time [ms]', fontsize=20)
    ylabel(r'$\nu_{a}$,$\nu_{b}$ [Hz]',fontsize=24)
    filename = str(figure_directory)+str(figure_counter)+str('_psth_deterministic.png')
    savefig(filename,bbox_inches='tight')
    figure_counter += 1


    figure()
    hist(out_mon_att.firing_rates(time_bin=180,mean=True),20,orientation='horizontal', normed=True, color='blue',label='pop A',linestyle='dashed',linewidth=3)
    ylim(0,np.max(mean_rate)+45)
    xlabel('P',fontsize=18)
    legend(loc='best')
    filename = str(figure_directory)+str(figure_counter)+str('_hist_a_deterministic.png')
    savefig(filename)     
    figure_counter += 1

   
    figure()
    hist(out_mon_att_1.firing_rates(time_bin=180,mean=True),20,orientation='horizontal', normed=True, color='green',label='pop B')
    ylim(0,np.max(mean_rate)+25)
    xlabel('P',fontsize=18)
    legend(loc='best')
    filename = str(figure_directory)+str(figure_counter)+str('_hist_b_deterministic.png')
    savefig(filename)     
    figure_counter += 1

    
    firing_rates = out_mon_att.firing_rates(time_bin=bins)
    index_att = np.nonzero(firing_rates>thr_att)
    attractor_indexes = np.array_split(index_att[0],np.where(np.diff(index_att[0])!=1)[0]+1)#find continuous element of the array
    raw_data_att = out_mon_att.sl.raw_data()  #get raw data

    firing_rates_att_1 = out_mon_att_1.firing_rates(time_bin=bins)
    index_att_1 = np.nonzero(firing_rates_att_1>thr_att)
    attractor_indexes_1 = np.array_split(index_att_1[0],np.where(np.diff(index_att_1[0])!=1)[0]+1)#find continuous element of the array
    raw_data_att_1 = out_mon_att_1.sl.raw_data()  #get raw data

    firing_rates_low = out_mon.firing_rates(time_bin=bins)
    index_low = np.nonzero(firing_rates_low<thr_low)
    low_indexes = np.array_split(index_low[0],np.where(np.diff(index_low[0])!=1)[0]+1)#find continuous element of the array
    raw_data_low = out_mon.sl.raw_data()  #get raw data

    firing_rates_all = out_mon.firing_rates(time_bin=bins)
    index_all_att = np.nonzero(firing_rates_all>thr_att)
    all_indexes = np.array_split(index_all_att[0],np.where(np.diff(index_all_att[0])!=1)[0]+1)#find continuous element of the array
    raw_data_all = out_mon.sl.raw_data()  #get raw data

    a1 = np.shape(attractor_indexes)
    #cicle over every attractor state and make a lists of isi
    sorted_spike_time = (raw_data_all[:][:,0])
    all_att_isi_all = []
    for i in range(a1[0]): 
        times_this_att = r_[attractor_indexes[i]]*60+np.min(sorted_spike_time)  
        if(len(times_this_att)>10):
            indexes = np.logical_and((sorted_spike_time >= times_this_att[0]), (sorted_spike_time< times_this_att[-1]))
            isi_this_phase = diff(sorted_spike_time[indexes])
            all_att_isi_all.extend(isi_this_phase)
    all_att_isi_all = np.array(all_att_isi_all)
    index_to_plot = np.where(all_att_isi_all > 0)
    all_att_isi_all = all_att_isi_all[index_to_plot]
    cv_att_all = std(all_att_isi_all)/mean(all_att_isi_all) 

    a1 = np.shape(attractor_indexes)
    #cicle over every attractor state and make a lists of isi
    sorted_spike_time = (raw_data_att[:][:,0])
    all_att_isi = []
    for i in range(a1[0]): 
        times_this_att = r_[attractor_indexes[i]]*60+np.min(sorted_spike_time)  
        if(len(times_this_att)>10):
            indexes = np.logical_and((sorted_spike_time >= times_this_att[0]), (sorted_spike_time< times_this_att[-1]))
            isi_this_phase = diff(sorted_spike_time[indexes])
            all_att_isi.extend(isi_this_phase)
    all_att_isi = np.array(all_att_isi)
    index_to_plot = np.where(all_att_isi > 0)
    all_att_isi = all_att_isi[index_to_plot]
    cv_att = std(all_att_isi)/mean(all_att_isi) 

    a1 = np.shape(attractor_indexes_1)
    #cicle over every attractor state and make a lists of isi
    sorted_spike_time = (raw_data_att_1[:][:,0])
    all_att_isi_1 = []
    for i in range(a1[0]): 
        times_this_att = r_[attractor_indexes_1[i]]*60+np.min(sorted_spike_time)  
        if(len(times_this_att)>10):
            indexes = np.logical_and((sorted_spike_time >= times_this_att[0]), (sorted_spike_time< times_this_att[-1]))
            isi_this_phase = diff(sorted_spike_time[indexes])
            all_att_isi_1.extend(isi_this_phase)
    all_att_isi_1 = np.array(all_att_isi_1)
    index_to_plot = np.where(all_att_isi_1 > 0)
    all_att_isi_1 = all_att_isi_1[index_to_plot]
    cv_att_1 = std(all_att_isi_1)/mean(all_att_isi_1) 

    sorted_spike_time_low = (raw_data_low[:][:,0])
    a1 = np.shape(low_indexes)
    all_low_isi = []
    for i in range(a1[0]): 
        times_this_low = r_[low_indexes[i]]*60+np.min(sorted_spike_time_low)  
        if(len(times_this_low)>5):
            indexes = np.logical_and((sorted_spike_time_low >= times_this_low[0]), (sorted_spike_time_low < times_this_low[-1]))
            isi_this_phase = diff(sorted_spike_time_low[indexes])
            all_low_isi.extend(isi_this_phase)
    all_low_isi = np.array(all_low_isi)
    index_to_plot = np.where(all_low_isi > 0)
    all_low_isi = all_low_isi[index_to_plot]
    cv_low = std(all_low_isi)/mean(all_low_isi) 

    #people go crazy for cv
    cv_string_att = "CV = %1.3f" % cv_att
    cv_string_att_1 = "CV = %1.3f" % cv_att_1
    cv_string_att_all = "CV = %1.3f" % cv_att_all
    if(std(all_low_isi) > 0):
        cv_string_low = "CV = %1.3f" % cv_low
        #text(mean(all_low_isi)+std(all_low_isi), 0.04, cv_string_low, family='serif', style='italic', ha='left', fontsize=18)

    #cv in function of shifts and delays
    from scipy import stats
    figure()
    (mu, sigma) = stats.expon.fit(all_att_isi) #Maximum Likelihood Estimate
    (x_r, phy_r, theta_r) = stats.recipinvgauss.fit(all_att_isi)
    n, bins, patches = plt.hist(all_att_isi, 60, normed=1, facecolor='blue', alpha=0.75)
    # add a 'best fit' line
    y = scipy.stats.expon.pdf( bins, mu, sigma)
    y_1 = scipy.stats.recipinvgauss.pdf(bins, x_r, phy_r, theta_r)
    #l = plot(bins, y, 'b--', linewidth=3, label='exp fit')
    ll = plot(bins, y_1, 'r--', linewidth=3)#, label='recipinvgauss fit')
    xlabel('ISI [ms]', fontsize=18)
    ylabel('P', fontsize=18)
    title('A up state: ISI distribution')
    legend(loc='upper right')
    #mu_exp_att = "mu = %1.3f" % mu
    #sigma_exp_att = "sigma = %1.3f" % sigma
    text(mean(all_att_isi)+std(all_att_isi), 0.0015, cv_string_att, family='serif', style='italic', ha='left', fontsize=18)
    filename = str(figure_directory)+str(figure_counter)+str('_popa_fit_deterministic.png')
    savefig(filename)     
    figure_counter += 1

    figure()
    (mu, sigma) = stats.expon.fit(all_att_isi_all) #Maximum Likelihood Estimate
    (x_r, phy_r, theta_r) = stats.recipinvgauss.fit(all_att_isi_all)
    n, bins, patches = plt.hist(all_att_isi_all, 60, normed=1, facecolor='yellow', alpha=0.75)
    # add a 'best fit' line
    y = scipy.stats.expon.pdf( bins, mu, sigma)
    y_1 = scipy.stats.recipinvgauss.pdf(bins, x_r, phy_r, theta_r)
    #l = plot(bins, y, 'b--', linewidth=3, label='exp fit')
    ll = plot(bins, y_1, 'r--', linewidth=3)#, label='recipinvgauss fit')
    xlabel('ISI [ms]', fontsize=18)
    ylabel('P', fontsize=18)
    title('combined up states: ISI distribution')
    legend(loc='upper right')
    #mu_exp_att = "mu = %1.3f" % mu
    #sigma_exp_att = "sigma = %1.3f" % sigma
    text(mean(all_att_isi_all)+std(all_att_isi_all), 0.0015, cv_string_att_all, family='serif', style='italic', ha='left', fontsize=18)
    filename = str(figure_directory)+str(figure_counter)+str('_popb_fit_deterministic.png')
    savefig(filename)     
    figure_counter += 1


def plot_balanced_double(d, bins=60, thr_att=10, thr_low=5):
    #analyze experiment balanced attractor 2

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
    out_mon_neu_z1 = np.load(d+'out_mon_neu_z1.pickle')



    raw_data = out_mon_att.sl.raw_data()
    mean_rate = out_mon_att.sl.mean_rates()
    t_start = np.min(raw_data[:,0])
    t_stop = np.max(raw_data[:,0])
    raw_data_att = out_mon_att.sl.raw_data()
    raw_data_att_1 = out_mon_att_1.sl.raw_data()

    #### LO ZOOM CAZZO@@@
    ttbin = 350
    fig, ax = plt.subplots(figsize=[5,4])
    ax.plot(np.linspace(t_start,t_stop,len(out_mon_att.firing_rates(time_bin=ttbin))), out_mon_att.firing_rates(time_bin=ttbin)*1.6, 'g', label='pop A')
    ax.plot(np.linspace(t_start,t_stop,len(out_mon_att_1.firing_rates(time_bin=ttbin))),out_mon_att_1.firing_rates(time_bin=ttbin)*1.65, 'b--', label='pop B')
    ax.plot(raw_data_att[:,0],raw_data_att[:,1],'g*', markersize=1, linewidth=2, alpha=0.4)
    ax.plot(raw_data_att_1[:,0],raw_data_att_1[:,1],'b*', markersize=1, linewidth=2, alpha=0.4)

    xticks([18000,200000,220000,240000,260000,280000], [0,2,4,6,8,10], fontsize=18)
    yticks(fontsize=18)
    xlabel('Time [s]', fontsize=18)
    ylabel('Freq [Hz]', fontsize=18)
    xlim([t_start, t_stop])
    legend(loc='best')
        
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    axins = zoomed_inset_axes(ax, 3, loc=4)
    axins.plot(raw_data_att[:,0],raw_data_att[:,1],'g*', markersize=3, linewidth=1, alpha=0.4)
    axins.tick_params(\
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off') # labels along the bottom edge are off
    x1, x2, y1, y2 = 261000,266500, 0, 20 
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    

    
    figure()
    hist(out_mon_att.firing_rates(time_bin=ttbin,mean=True)*1.6,20,orientation='horizontal', normed=True, color='green', label='pop A')
    xlabel('norm. counts',fontsize=18)
    legend(loc='best')
    xlim([0.0,0.03])
    ylim([0,100])
    figure()
    hist(out_mon_att_1.firing_rates(time_bin=ttbin,mean=True)*1.65,20,orientation='horizontal', normed=True, color='blue', label='pop B')
    xlabel('norm. counts',fontsize=18)
    xlim([0.0,0.02])
    ylim([0,100])
    legend(loc='best')


    pyNCS.monitors.MeanRatePlot([out_mon_att,out_mon_att_1],time_bin=150)
    legend(["pop A","pop B"], loc=1)

    xlim(t_start,t_stop)
    ylim(0,np.max(mean_rate)+45)
    xlabel('Time [ms]', fontsize=22)
    ylabel('Mean Firing Rate [Hz]', fontsize=22)
    filename = str(figure_directory)+str(figure_counter)+str('_mean_rate.eps')
    savefig(filename,format='eps')     
    figure_counter +=1

    figure()
    raw_data_att = out_mon_att.sl.raw_data()
    raw_data_att_1 = out_mon_att_1.sl.raw_data()
    
    plot(raw_data_att[:,0],raw_data_att[:,1],'b*', markersize=3, linewidth=2, alpha=0.4)
    plot(raw_data_att_1[:,0],raw_data_att_1[:,1],'g*', markersize=3, linewidth=2, alpha=0.4)
    legend(["pop A","pop B"], loc=1)
    xlim(t_start,t_stop)
    ylim(0,64)
    xlabel('Time [ms]', fontsize=18)
    ylabel('Neuron Id', fontsize=18)
    filename = str(figure_directory)+str(figure_counter)+str('_mean_rate.eps')
    savefig(filename,format='eps')     
    figure_counter += 1

    figure()
    hist(out_mon_att.firing_rates(time_bin=30,mean=True),20,orientation='horizontal', normed=True, color='blue')
    ylim(0,np.max(mean_rate)+45)
    xlabel('P',fontsize=18)
    title('pop A')
    filename = str(figure_directory)+str(figure_counter)+str('_hist_hor.eps')
    savefig(filename,format='eps')     
    figure_counter += 1

    figure()
    hist(out_mon_att_1.firing_rates(time_bin=60,mean=True),20,orientation='horizontal', normed=True, color='green')
    ylim(0,np.max(mean_rate)+45)
    xlabel('P',fontsize=18)
    title('pop B')
    filename = str(figure_directory)+str(figure_counter)+str('_hist_hor.eps')
    savefig(filename,format='eps')     
    figure_counter += 1

    firing_rates = out_mon_att.firing_rates(time_bin=bins)
    index_att = np.nonzero(firing_rates>thr_att)
    attractor_indexes = np.array_split(index_att[0],np.where(np.diff(index_att[0])!=1)[0]+1)#find continuous element of the array
    raw_data_att = out_mon_att.sl.raw_data()  #get raw data

    firing_rates_att_1 = out_mon_att_1.firing_rates(time_bin=bins)
    index_att_1 = np.nonzero(firing_rates_att_1>thr_att)
    attractor_indexes_1 = np.array_split(index_att_1[0],np.where(np.diff(index_att_1[0])!=1)[0]+1)#find continuous element of the array
    raw_data_att_1 = out_mon_att_1.sl.raw_data()  #get raw data

    firing_rates_low = out_mon.firing_rates(time_bin=bins)
    index_low = np.nonzero(firing_rates_low<thr_low)
    low_indexes = np.array_split(index_low[0],np.where(np.diff(index_low[0])!=1)[0]+1)#find continuous element of the array
    raw_data_low = out_mon.sl.raw_data()  #get raw data

    firing_rates_all = out_mon.firing_rates(time_bin=bins)
    index_all_att = np.nonzero(firing_rates_all>thr_att)
    all_indexes = np.array_split(index_all_att[0],np.where(np.diff(index_all_att[0])!=1)[0]+1)#find continuous element of the array
    raw_data_all = out_mon.sl.raw_data()  #get raw data

    firing_rates_neu_z = out_mon_neu_z.firing_rates(time_bin=bins)
    index_neu_z = np.nonzero(firing_rates_neu_z>thr_att)
    neu_z_indexes = np.array_split(index_neu_z[0],np.where(np.diff(index_neu_z[0])!=1)[0]+1)#find continuous element of the array
    raw_data_neu_z = out_mon_neu_z.sl.raw_data()  #get raw data

    firing_rates_neu_z1 = out_mon_neu_z1.firing_rates(time_bin=bins)
    index_neu_z1 = np.nonzero(firing_rates_neu_z1>thr_att)
    neu_z1_indexes = np.array_split(index_neu_z[0],np.where(np.diff(index_neu_z1[0])!=1)[0]+1)#find continuous element of the array
    raw_data_neu_z1 = out_mon_neu_z1.sl.raw_data()  #get raw data

    a1 = np.shape(all_indexes)
    #cicle over every attractor state and make a lists of isi
    sorted_spike_time = (raw_data[:][:,0])
    all_att_isi_all = []
    for i in range(a1[0]): 
        times_this_att = r_[all_indexes[i]]*60+np.min(sorted_spike_time)  
    if(len(times_this_att)>10):
        indexes = np.logical_and((sorted_spike_time >= times_this_att[0]), (sorted_spike_time< times_this_att[-1]))
        isi_this_phase = diff(sorted_spike_time[indexes])
        all_att_isi_all.extend(isi_this_phase)
    all_att_isi_all = np.array(all_att_isi_all)
    index_to_plot = np.where(all_att_isi_all > 0)
    all_att_isi_all = all_att_isi_all[index_to_plot]
    cv_att_all = std(all_att_isi_all)/mean(all_att_isi_all) 

    a1 = np.shape(neu_z_indexes)
    #cicle over every attractor state and make a lists of isi
    sorted_spike_time = (raw_data_neu_z[:][:,0])
    neu_z_att_isi = []
    spike_raster_neu_z = []
    for i in range(a1[0]): 
        times_this_att = r_[neu_z_indexes[i]]*60+np.min(sorted_spike_time)  
    if(len(times_this_att)>10):
        indexes = np.logical_and((sorted_spike_time >= times_this_att[0]), (sorted_spike_time< times_this_att[-1]))
        spike_raster_neu_z.append(sorted_spike_time[indexes])
        isi_this_phase = diff(sorted_spike_time[indexes])
        neu_z_att_isi.extend(isi_this_phase)
    neu_z_att_isi = np.array(neu_z_att_isi)
    index_to_plot = np.where(neu_z_att_isi > 0)
    neu_z_att_isi = neu_z_att_isi[index_to_plot]
    if(len(neu_z_att_isi) > 0 ):
        neu_z_att_isi = neu_z_att_isi-min(neu_z_att_isi)
        cv_neu_z = std(neu_z_att_isi)/mean(neu_z_att_isi)     
    else:
        neu_z_att_isi = [0]
        cv_neu_z = 0 
   
    a1 = np.shape(neu_z1_indexes)
    #cicle over every attractor state and make a lists of isi
    sorted_spike_time = (raw_data_neu_z1[:][:,0])
    neu_z1_att_isi = []
    spike_raster_neu_z1 = []
    for i in range(a1[0]): 
        times_this_att = r_[neu_z1_indexes[i]]*60+np.min(sorted_spike_time)  
        if(len(times_this_att)>10):
            indexes = np.logical_and((sorted_spike_time >= times_this_att[0]), (sorted_spike_time< times_this_att[-1]))
            spike_raster_neu_z1.append(sorted_spike_time[indexes])
            isi_this_phase = diff(sorted_spike_time[indexes])
            neu_z1_att_isi.extend(isi_this_phase)
    neu_z1_att_isi = np.array(neu_z1_att_isi)
    index_to_plot = np.where(neu_z1_att_isi > 0)
    neu_z1_att_isi = neu_z1_att_isi[index_to_plot]
    if(len(neu_z1_att_isi) > 0 ):
        neu_z1_att_isi = neu_z1_att_isi-min(neu_z1_att_isi)
        cv_neu_z1 = std(neu_z1_att_isi)/mean(neu_z1_att_isi)     
    else:
        neu_z1_att_isi = [0]
        cv_neu_z1 = 0 


    #people go crazy for cv
    cv_string_neu_z = "CV = %1.3f" % cv_neu_z
    cv_string_neu_z1 = "CV = %1.3f" % cv_neu_z1

    #cv in function of shifts and delays
    from scipy import stats

    if(len(neu_z_att_isi) > 0):
        figure()
        (mu, sigma) = stats.expon.fit(neu_z_att_isi) #Maximum Likelihood Estimate
        (x_r, phy_r, theta_r) = stats.recipinvgauss.fit(neu_z_att_isi)
        n, bins, patches = plt.hist(neu_z_att_isi, 60, normed=1, facecolor='blue', alpha=0.75)
        # add a 'best fit' line
        y = scipy.stats.expon.pdf( bins, mu, sigma)
        zero_fit_index = np.nonzero(y==0)
        z1,z2 = np.shape(zero_fit_index)
        y_1 = scipy.stats.recipinvgauss.pdf(bins, x_r, phy_r, theta_r)
        #l = plot(bins[2:], y[2:], 'r--', linewidth=3, label='exp fit')
        #ll = plot(bins, y_1, 'r--', linewidth=3, label='recipinvgauss fit')
        xlabel('ISI [ms]', fontsize=22)
        ylabel('P', fontsize=22)
        title('single neuron up states: ISI distribution', fontsize=22)
        legend(loc='upper right')
        mu_exp_att = "mu = %1.3f" % mu
        sigma_exp_att = "sigma = %1.3f" % sigma
        text(mean(neu_z_att_isi)+std(neu_z_att_isi),max(n), cv_string_neu_z, family='serif', style='italic', ha='left', fontsize=22)
        filename = str(figure_directory)+str(figure_counter)+str('_hist_fit_combined_up.eps')
        savefig(filename,format='eps')     
        figure_counter += 1

    if(len(neu_z1_att_isi) > 0):
        figure()
        (mu, sigma) = stats.expon.fit(neu_z1_att_isi) #Maximum Likelihood Estimate
        (x_r, phy_r, theta_r) = stats.recipinvgauss.fit(neu_z1_att_isi)
        n, bins, patches = plt.hist(neu_z1_att_isi, 60, normed=1, facecolor='green', alpha=0.75)
        # add a 'best fit' line
        y = scipy.stats.expon.pdf( bins, mu, sigma)
        zero_fit_index = np.nonzero(y==0)
        z1,z2 = np.shape(zero_fit_index)
        #y_1 = scipy.stats.recipinvgauss.pdf(bins, x_r, phy_r, theta_r)
        #l = plot(bins[2:], y[2:], 'r--', linewidth=3, label='exp fit')
        #ll = plot(bins, y_1, 'r--', linewidth=3, label='recipinvgauss fit')
        xlabel('ISI [ms]', fontsize=22)
        ylabel('P', fontsize=22)
        title('single neuron up states: ISI distribution', fontsize=22)
        legend(loc='upper right')
        mu_exp_att = "mu = %1.3f" % mu
        sigma_exp_att = "sigma = %1.3f" % sigma
        text(mean(neu_z1_att_isi)+std(neu_z1_att_isi),max(n), cv_string_neu_z1, family='serif', style='italic', ha='left', fontsize=22)
        filename = str(figure_directory)+str(figure_counter)+str('_hist_fit_combined_up.eps')
        savefig(filename,format='eps')     
        figure_counter += 1


def plot_balanced_single(d, bins=60, thr_att=10, thr_low=5):
	#analyze experiment balanced attractor 1

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

	pyNCS.monitors.MeanRatePlot([out_mon_att],time_bin=150)
	legend(["pop A"], loc=1)
	hold(True)
	raw_data_att = out_mon_att.sl.raw_data()
	plot(raw_data_att[:,0],raw_data_att[:,1],'b*', markersize=3, linewidth=2, alpha=0.4)

	xlim(t_start,t_stop)
	ylim(0,np.max(mean_rate)+45)
	xlabel('Time [ms]', fontsize=22)
	ylabel('Mean Firing Rate [Hz]', fontsize=22)
	filename = str(figure_directory)+str(figure_counter)+str('_mean_rate.png')
	savefig(filename)     
	figure_counter +=1

	figure()
	raw_data_att = out_mon_att.sl.raw_data()
	plot(raw_data_att[:,0],raw_data_att[:,1],'b*', markersize=3, linewidth=2, alpha=0.4)
	legend(["pop A"],loc=1)
	xlim(t_start,t_stop)
	ylim(0,64)
	xlabel('Time [ms]', fontsize=18)
	ylabel('Neuron Id', fontsize=18)
	filename = str(figure_directory)+str(figure_counter)+str('_mean_rate.png')
	savefig(filename)     
	figure_counter += 1


	figure()
	hist(out_mon_att.firing_rates(time_bin=60,mean=True),20,orientation='horizontal', normed=True, color='blue')
	ylim(0,np.max(mean_rate)+45)
	xlabel('P',fontsize=18)
	title('pop A')
	filename = str(figure_directory)+str(figure_counter)+str('_hist_hor.png')
	savefig(filename)     
	figure_counter += 1

	firing_rates = out_mon_att.firing_rates(time_bin=bins)
	index_att = np.nonzero(firing_rates>thr_att)
	attractor_indexes = np.array_split(index_att[0],np.where(np.diff(index_att[0])!=1)[0]+1)#find continuous element of the array
	raw_data_att = out_mon_att.sl.raw_data()  #get raw data

	firing_rates_low = out_mon.firing_rates(time_bin=bins)
	index_low = np.nonzero(firing_rates_low<thr_low)
	low_indexes = np.array_split(index_low[0],np.where(np.diff(index_low[0])!=1)[0]+1)#find continuous element of the array
	raw_data_low = out_mon.sl.raw_data()  #get raw data

	firing_rates_all = out_mon.firing_rates(time_bin=bins)
	index_all_att = np.nonzero(firing_rates_all>thr_att)
	all_indexes = np.array_split(index_all_att[0],np.where(np.diff(index_all_att[0])!=1)[0]+1)#find continuous element of the array
	raw_data_all = out_mon.sl.raw_data()  #get raw data

	firing_rates_neu_z = out_mon_neu_z.firing_rates(time_bin=bins)
	index_neu_z = np.nonzero(firing_rates_neu_z>thr_att)
	neu_z_indexes = np.array_split(index_neu_z[0],np.where(np.diff(index_neu_z[0])!=1)[0]+1)#find continuous element of the array
	raw_data_neu_z = out_mon_neu_z.sl.raw_data()  #get raw data

	a1 = np.shape(all_indexes)
	#cicle over every attractor state and make a lists of isi
	sorted_spike_time = (raw_data[:][:,0])
	all_att_isi_all = []
	for i in range(a1[0]): 
		times_this_att = r_[all_indexes[i]]*60+np.min(sorted_spike_time)  
		if(len(times_this_att)>10):
		    indexes = np.logical_and((sorted_spike_time >= times_this_att[0]), (sorted_spike_time< times_this_att[-1]))
		    isi_this_phase = diff(sorted_spike_time[indexes])
		    all_att_isi_all.extend(isi_this_phase)
	all_att_isi_all = np.array(all_att_isi_all)
	index_to_plot = np.where(all_att_isi_all > 0)
	all_att_isi_all = all_att_isi_all[index_to_plot]
	cv_att_all = std(all_att_isi_all)/mean(all_att_isi_all) 

	a1 = np.shape(neu_z_indexes)
	#cicle over every attractor state and make a lists of isi
	sorted_spike_time = (raw_data_neu_z[:][:,0])
	neu_z_att_isi = []
	spike_raster_neu_z = []
	for i in range(a1[0]): 
		times_this_att = r_[neu_z_indexes[i]]*60+np.min(sorted_spike_time)  
		if(len(times_this_att)>10):
		    indexes = np.logical_and((sorted_spike_time >= times_this_att[0]), (sorted_spike_time< times_this_att[-1]))
		    spike_raster_neu_z.append(sorted_spike_time[indexes])
		    isi_this_phase = diff(sorted_spike_time[indexes])
		    neu_z_att_isi.extend(isi_this_phase)
	neu_z_att_isi = np.array(neu_z_att_isi)
	index_to_plot = np.where(neu_z_att_isi > 0)
	neu_z_att_isi = neu_z_att_isi[index_to_plot]
	neu_z_att_isi = neu_z_att_isi-min(neu_z_att_isi)
	cv_neu_z = std(neu_z_att_isi)/mean(neu_z_att_isi) 

	z1 = np.shape(spike_raster_neu_z)
	figure()
	#cicle over all spike time and align them to zero and plot the raster
	for i in range(int(r_[z1])):
		spike_raster_neu_z[i] = spike_raster_neu_z[i] - min(spike_raster_neu_z[i])
		y_ax = np.repeat(i,len(spike_raster_neu_z[i]))
		plot(spike_raster_neu_z[i],y_ax, 'b*', markersize=5, linewidth=2, alpha=0.4)
		hold(True)
	xlabel('time [ms]', fontsize=18)
	ylabel('trial ID', fontsize=18)
	filename = str(figure_directory)+str(figure_counter)+str('_raster_single_neu_pop_a_trial.png')
	savefig(filename)     
	figure_counter += 1

	#people go crazy for cv
	cv_string_neu_z = "CV = %1.3f" % cv_neu_z

	#cv in function of shifts and delays
	from scipy import stats

	figure()
	(mu, sigma) = stats.expon.fit(neu_z_att_isi) #Maximum Likelihood Estimate
	(x_r, phy_r, theta_r) = stats.recipinvgauss.fit(neu_z_att_isi)
	n, bins, patches = plt.hist(neu_z_att_isi, 60, normed=1, facecolor='blue', alpha=0.75)
	# add a 'best fit' line
	y = scipy.stats.expon.pdf( bins, mu, sigma)
	zero_fit_index = np.nonzero(y==0)
	z1,z2 = np.shape(zero_fit_index)
	y_1 = scipy.stats.recipinvgauss.pdf(bins, x_r, phy_r, theta_r)
	#l = plot(bins[2:], y[2:], 'r--', linewidth=3, label='exp fit')
	#ll = plot(bins, y_1, 'r--', linewidth=3, label='recipinvgauss fit')
	xlabel('ISI [ms]', fontsize=22)
	ylabel('P', fontsize=22)
	title('single neuron up states: ISI distribution', fontsize=22)
	legend(loc='upper right')
	mu_exp_att = "mu = %1.3f" % mu
	sigma_exp_att = "sigma = %1.3f" % sigma
	text(mean(neu_z_att_isi)+std(neu_z_att_isi),max(n)-max(n/10), cv_string_neu_z, family='serif', style='italic', ha='left', fontsize=22)
	filename = str(figure_directory)+str(figure_counter)+str('_hist_fit_combined_up.png')
	savefig(filename)     
	figure_counter += 1
