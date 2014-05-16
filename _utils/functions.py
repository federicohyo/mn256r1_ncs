import numpy as np
from pylab import *
import os as os

def ismember(a, b):
    # tf = np.in1d(a,b) # for newer versions of numpy
    tf = np.array([i in b for i in a])
    u = np.unique(a[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
    return tf, index

#### Mean firing rate of each neuron
def mean_neu_firing(spike_train, n_neurons,simulation_time):
    import numpy as np
    mean_rate   = np.zeros([len(n_neurons)])
    #simulation_time = [np.min(spike_train[0][:]), np.max(spike_train[0][:])]
    for i in range(len(n_neurons)):
        index_neu = np.where(np.logical_and(spike_train[:,1] == n_neurons[i], np.logical_and(spike_train[:,0] > simulation_time[0] , spike_train[:,0] < simulation_time[1] )) )
        mean_rate[i] = len(index_neu[0])*1000.0/(simulation_time[1]-simulation_time[0]) # time unit: ms
    return mean_rate

