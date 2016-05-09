import numpy as np
from pylab import *
import os as os
import pyNCS

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
        index_neu = np.where(np.logical_and(spike_train[1][:] == n_neurons[i], np.logical_and(spike_train[0][:] > simulation_time[0] , spike_train[0][:] < simulation_time[1] )) )
        mean_rate[i] = len(index_neu[0])*1000.0/(simulation_time[1]-simulation_time[0]) # time unit: ms
    return mean_rate

def generate_bias(n_neu, min_rate, max_rate):
    neu_biases = min_rate + (max_rate - min_rate) * random(n_neu)
    return neu_biases

def compute_decoders_1(x_values, A, nsteps, function=lambda x:x):
    
    # get the desired decoded value for each sample point
    value=np.array([function(x) for x in x_values])
    A = np.array(A)
    
    # find the optimum linear decoder
    #A=np.array(A)#.T
    Gamma=np.dot(A, A.T)
    Upsilon=np.dot(A, value)
    Ginv=np.linalg.pinv(Gamma)        
    decoder=np.dot(Ginv,Upsilon)
    return decoder

#find optimal linear decoder
def compute_decoders(x_values, A, nsteps,function=lambda x:x):

    #function to be encoded  
    value=np.array([[function(x)] for x in x_values]) 
    value=np.reshape(value,nsteps)

    # find the optimum linear decoder
    A=np.array(A)#.T
    gamma=np.dot(A, A.T) #I diagonal noise add I = eye(len(gamma))
    upsilon=np.dot(A, value) 
    ginv=np.linalg.pinv(gamma)        
    decoder=np.dot(ginv,upsilon) # i do not have dt in the denumerator ?
    
    return decoder

# compute the tuning curves for a population    
def compute_tuning_curves_sram(neuron_ids, min_rate,max_rate, min_rate_fixed, max_rate_fixed, nsteps, setup, chip, mypop, inputpop, do_plot = 0, pop_name = 'A'):

    n_neu = len(neuron_ids)
    neu_biases = generate_bias(n_neu, min_rate, max_rate)
    rate_biased = np.repeat(neu_biases,1)
    rate_biased = r_[[rate_biased]*nsteps]
    x_values = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)
    stim_fixed = r_[[np.linspace(min_rate_fixed,max_rate_fixed,nsteps)]*n_neu*1]
    rate_fixed = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)
    stim_matrix = stim_fixed+rate_biased.T
    timebins = np.linspace(0, 8000,nsteps)

    tf,index = ismember(inputpop.soma.laddr,neuron_ids)
    index_ok_w = inputpop[tf].synapses['sram_exc'].addr['weight'] == 8 # we only get one weight middle
    index_ok_syn = inputpop[tf].synapses['sram_exc'].addr['syn'] == 1
    index_f_ok = np.where(index_ok_w & index_ok_syn)
    islearning = inputpop[tf].synapses['sram_exc'][index_f_ok]
    spiketrain = islearning.spiketrains_inh_poisson(stim_matrix,timebins)

    out_mon = pyNCS.monitors.SpikeMonitor(mypop.soma)
    in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)  
    setup.monitors.import_monitors([out_mon, in_mon])

    x = setup.stimulate(spiketrain, send_reset_event=False, tDuration=max(timebins))

    # A matrix 
    n_neu = neuron_ids
    AN = out_mon.sl.firing_rate(1.*max(timebins)/nsteps)
    a1, a2 = np.shape(AN)
    if a2 == nsteps:
        A = [a[::] for a in AN[n_neu.astype(int),:]]
    else:    
        A = [a[:-1:] for a in AN[n_neu.astype(int),:]]   
    A[:len(n_neu)/2] = [a[::-1] for a in A[:len(n_neu)/2]] #flippo A uno ogni 2 non so come ma fabio lo sa, ho capito

    if do_plot == 1:
        figure()
        count = 0
        a1, a2 = np.shape(A)
        for p in A:
            count += 1
            if a2 == nsteps:
                if( count <= len(neuron_ids)/2):
                    plot(rate_fixed,p,'mo-') #add color b
                else:
                    plot(rate_fixed,p,'co-') #add color r
            else:
                if( count <= len(neuron_ids)/2):
                    plot(rate_fixed[:-1:],p,'mo-') #add color b
                else:
                    plot(rate_fixed[:-1:],p,'co-') #add color r
        xlabel(r'$\nu_{in} [Hz]$',fontsize=18)
        ylabel(r'$\nu_{out} [Hz]$',fontsize=18)
        string_title = ('Tuning curves for population ' +  pop_name)
        title(string_title)
        show()

    #function to be encoded  
    x_values=[i*2.0/nsteps - 1 for i in range(nsteps)]
    value=np.array([[function(x)] for x in x_values])
    value=np.reshape(value,nsteps)                

    return x_values, A, neu_biases 



def compute_phy_neurons_pop(neuron_ids, min_rate,max_rate, min_rate_fixed, max_rate_fixed, nsteps, setup, chip, mypop, inputpop, do_plot = 0, pop_name = 'A'):

    import random    

    # generate a stim 
    x_values = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)

    n_neu = len(neuron_ids)

    # compute input corresponding to x
    # since we will use 7 synapses for this input we multiply neuron id *7
    input_matrix = r_[[np.linspace(min_rate_fixed,max_rate_fixed,nsteps)]*n_neu*2]
    step = 0
    #for x in x_values:
    #    for i in range(len(neuron_ids)):
    #        input_matrix[i*7:7*(i+1),step] = np.repeat(x,7)  
    #    step += 1

    timebins = np.linspace(0, 8000,nsteps)

    tf,index = ismember(inputpop.soma.laddr,neuron_ids)
    index_ok_w = inputpop[tf].synapses['sram_exc'].addr['weight'] == 12 # we only get one weight middle
    index_ok_syn = inputpop[tf].synapses['sram_exc'].addr['syn'] == 0
    index_f_ok = np.where(index_ok_w & index_ok_syn)
    index_f_ok = index_ok_w   
    islearning = inputpop[tf].synapses['sram_exc'][index_f_ok]

    spiketrain = islearning.spiketrains_inh_poisson(input_matrix,timebins)

    tf,index = ismember(mypop.soma.laddr,neuron_ids)
    out_mon = pyNCS.monitors.SpikeMonitor(mypop[tf].soma)
    in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)  
    setup.monitors.import_monitors([out_mon, in_mon])

    x = setup.stimulate(spiketrain, send_reset_event=False, tDuration=max(timebins))

    # A matrix 
    n_neu = neuron_ids
    AN = out_mon.sl.firing_rate(1.*max(timebins)/nsteps)
    a1, a2 = np.shape(AN)

    x_values_array = np.array(x_values)
    indexes = np.argsort(x_values_array)
    if do_plot == 1:
		figure()
		count = 0
		a1, a2 = np.shape(AN)
		for p in AN:
			count += 1
			if( count%2):
				plot(x_values_array[indexes],p[indexes],'mo-', label='moscap') #add color b
			else:
				plot(x_values_array[indexes],p[indexes],'co-', label='cmim') #add color r
		xlabel(r'$\nu_{in} [Hz]$',fontsize=20)
		ylabel(r'$\nu_{out} [Hz]$',fontsize=20)
		string_title = ('single neuron transfer function')
		title(string_title)
		#legend()

		moscap = AN[0:21:]
		n_neu_moscap, s_step_moscap = np.shape(AN[0::2])
		moscap = np.array(moscap)
		freq_mean_moscap = np.sum(moscap,axis=0) / n_neu_moscap

		cmim = AN[28:49:]
		n_neu_cmim, s_step_cmim = np.shape(AN[1::2])
		cmim = np.array(cmim)
		freq_mean_cmim = np.sum(cmim,axis=0) / n_neu_cmim

		mean_error_moscap = []
		for i in range(s_step_moscap):
			mean_error_moscap.append(np.std(moscap[:,i]))

		mean_error_cmim = []
		for i in range(s_step_cmim):
			mean_error_cmim.append(np.std(cmim[:,i]))

		stim = np.linspace(min_rate_fixed,max_rate_fixed,len(freq_mean_moscap))

		figure()
		plot(stim[:-1:],freq_mean_moscap[:-1:], 'bo--', label='pop A')
		fill_between(stim[:-1:], freq_mean_moscap[:-1:]-mean_error_moscap[:-1:], freq_mean_moscap[:-1:]+mean_error_moscap[:-1:],alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=2, antialiased=True)
		hold(True)
		plot(stim[:-1:],freq_mean_cmim[:-1:], 'go-', label='pop B')
		fill_between(stim[:-1:], freq_mean_cmim[:-1:]-mean_error_cmim[:-1:], freq_mean_cmim[:-1:]+mean_error_cmim[:-1:],alpha=0.3, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=2)
		ylim(0,np.max(freq_mean_moscap[:-1:])+30)
		xlabel(r'$\nu_{in} [Hz]$',fontsize=22)
		ylabel(r'$\nu_{out} [Hz]$',fontsize=22)
		#yticks(np.linspace(0,200,9))
		#xticks(np.linspace(0,250,11))
		#string_title = ('single neuron transfer function')
		#title(string_title)
		legend(loc='upper left')
		#show()

    return x_values, AN


def compute_phy_neurons(neuron_ids, min_rate,max_rate, min_rate_fixed, max_rate_fixed, nsteps, setup, chip, mypop, inputpop, do_plot = 0, pop_name = 'A'):

    import random    

    # generate a stim 
    x_values = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)

    n_neu = len(neuron_ids)

    # compute input corresponding to x
    # since we will use 7 synapses for this input we multiply neuron id *7
    input_matrix = np.zeros([len(neuron_ids)*7,nsteps])
    step = 0
    for x in x_values:
        for i in range(len(neuron_ids)):
            input_matrix[i*7:7*(i+1),step] = np.repeat(x,7)  
        step += 1

    timebins = np.linspace(0, 8000,nsteps)

    tf,index = ismember(inputpop.soma.laddr,neuron_ids)
    isinput_exc = inputpop[tf].synapses['sram_exc']
    spiketrain = isinput_exc.spiketrains_inh_poisson(input_matrix,timebins)

    tf,index = ismember(mypop.soma.laddr,neuron_ids)
    out_mon = pyNCS.monitors.SpikeMonitor(mypop[tf].soma)
    in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)  
    setup.monitors.import_monitors([out_mon, in_mon])

    x = setup.stimulate(spiketrain, send_reset_event=False, tDuration=max(timebins))

    # A matrix 
    n_neu = neuron_ids
    AN = out_mon.sl.firing_rate(1.*max(timebins)/nsteps)
    a1, a2 = np.shape(AN)

    x_values_array = np.array(x_values)
    indexes = np.argsort(x_values_array)
    if do_plot == 1:
		figure()
		count = 0
		a1, a2 = np.shape(AN)
		for p in AN:
			count += 1
			if( count%2):
				plot(x_values_array[indexes],p[indexes],'mo-', label='moscap') #add color b
			else:
				plot(x_values_array[indexes],p[indexes],'co-', label='cmim') #add color r
		xlabel(r'$\nu_{in} [Hz]$',fontsize=20)
		ylabel(r'$\nu_{out} [Hz]$',fontsize=20)
		string_title = ('single neuron transfer function')
		title(string_title)
		#legend()

		moscap = AN[0::2]
		n_neu_moscap, s_step_moscap = np.shape(AN[0::2])
		moscap = np.array(moscap)
		freq_mean_moscap = np.sum(moscap,axis=0) / n_neu_moscap

		cmim = AN[1::2]
		n_neu_cmim, s_step_cmim = np.shape(AN[1::2])
		cmim = np.array(cmim)
		freq_mean_cmim = np.sum(cmim,axis=0) / n_neu_cmim

		mean_error_moscap = []
		for i in range(s_step_moscap):
			mean_error_moscap.append(np.std(moscap[:,i]))

		mean_error_cmim = []
		for i in range(s_step_cmim):
			mean_error_cmim.append(np.std(cmim[:,i]))

		stim = np.linspace(min_rate_fixed,max_rate_fixed,len(freq_mean_moscap))

		figure()
		plot(stim[:-1:],freq_mean_moscap[:-1:], 'bo-', label='moscap')
		fill_between(stim[:-1:], freq_mean_moscap[:-1:]-mean_error_moscap[:-1:], freq_mean_moscap[:-1:]+mean_error_moscap[:-1:],alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=2, antialiased=True)
		hold(True)
		plot(stim[:-1:],freq_mean_cmim[:-1:], 'go-', label='cmim')
		fill_between(stim[:-1:], freq_mean_cmim[:-1:]-mean_error_cmim[:-1:], freq_mean_cmim[:-1:]+mean_error_cmim[:-1:],alpha=0.3, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=2)
		ylim(0,np.max(freq_mean_moscap[:-1:])+30)
		xlabel(r'$\nu_{in} [Hz]$',fontsize=20)
		ylabel(r'$\nu_{out} [Hz]$',fontsize=20)
		string_title = ('single neuron transfer function')
		title(string_title)
		legend(loc='upper left')
		show()

    return x_values, AN


def compute_tuning_curves_sram_1(encoders,neuron_ids, min_rate,max_rate, min_rate_fixed, max_rate_fixed, nsteps, setup, chip, mypop, inputpop, do_plot = 0, pop_name = 'A'):
    dimensions = encoders.shape[1]

    import random    

    # generate a set of x values to sample at
    # for us -1 is min_rate_fixed and +1 is max_rate_fixed
    if dimensions==1:
        #x_values = numpy.array([[i*2.0/N_samples - 1.0] for i in range(N_samples)])
        x_values = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)
        x_values = np.reshape(x_values,[len(x_values),1])
        #x_values = [random.random()*(max_rate_fixed-min_rate_fixed) for i in range(nsteps)]
    else:
        #x_values = [np.linspace(min_rate_fixed,max_rate_fixed,nsteps),np.linspace(min_rate_fixed,max_rate_fixed,nsteps)] 
        #x_values = [[random.random()*(max_rate_fixed-min_rate_fixed),random.random()*(max_rate_fixed-min_rate_fixed)] for i in range(nsteps)]
        y = [random.random()*(max_rate_fixed-min_rate_fixed) for i in range(nsteps)]
        y=np.array(y)
        y=y.T 
        x_values = [np.linspace(min_rate_fixed,max_rate_fixed,nsteps),y] 
        x_values = np.array(x_values)
        x_values = x_values.T
        #norm = np.sqrt(np.sum(x_values*x_values, axis=1))
        #x_values = x_values / norm[:,None]
        #scale = np.random.uniform(size=nsteps)**(1.0/dimensions)
        #x_values = x_values * scale[:,None]


    n_neu = len(neuron_ids)
    neu_biases = generate_bias(n_neu, min_rate, max_rate)

    # compute input corresponding to x
    # since we will use 7 synapses for this input we multiply neuron id *7
    input_matrix = np.zeros([len(neuron_ids)*7,nsteps])
    step = 0
    for x in x_values:
        for i in range(len(neuron_ids)):
            input_matrix[i*7:7*(i+1),step] = np.repeat((np.dot(x,encoders[i])+neu_biases[i]),7)  
        step += 1

    timebins = np.linspace(0, 8000,nsteps)

    tf,index = ismember(inputpop.soma.laddr,neuron_ids)
    isinput_exc = inputpop[tf].synapses['sram_exc']
    spiketrain = isinput_exc.spiketrains_inh_poisson(input_matrix,timebins)

    out_mon = pyNCS.monitors.SpikeMonitor(mypop.soma)
    in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)  
    setup.monitors.import_monitors([out_mon, in_mon])

    x = setup.stimulate(spiketrain, send_reset_event=False, tDuration=max(timebins))

    # A matrix 
    n_neu = neuron_ids
    AN = out_mon.sl.firing_rate(1.*max(timebins)/nsteps)
    a1, a2 = np.shape(AN)
    if a2 == nsteps:
        A = [a[::] for a in AN[n_neu.astype(int),:]]
    else:    
        A = [a[:-1:] for a in AN[n_neu.astype(int),:]]   
    #A[:len(n_neu)/2] = [a[::-1] for a in A[:len(n_neu)/2]] 

    x_values_array = np.array(x_values)
    indexes = np.argsort(x_values_array)
    if do_plot == 1:
        figure()
        count = 0
        a1, a2 = np.shape(A)
        for p in A:
            count += 1
            if a2 == nsteps:
                if( count <= len(neuron_ids)/2):
                    plot(x_values_array[indexes],p[indexes],'mo-') #add color b
                else:
                    plot(x_values_array[indexes],p[indexes],'co-') #add color r
            else:
                if( count <= len(neuron_ids)/2):
                    plot(x_values_array[indexes][:-1:],p[indexes],'mo-') #add color b
                else:
                    plot(x_values_array[indexes][:-1:],p[indexes],'co-') #add color r
        xlabel(r'$\nu_{in} [Hz]$',fontsize=18)
        ylabel(r'$\nu_{out} [Hz]$',fontsize=18)
        string_title = ('Tuning curves for population ' +  pop_name)
        title(string_title)
        show()

    return x_values, A, neu_biases

# compute the tuning curves for a population    
def compute_tuning_curves(neuron_ids, min_rate,max_rate, min_rate_fixed, max_rate_fixed, nsteps, setup, chip, mypop, inputpop, do_plot = 0, pop_name = 'A'):

    n_neu = len(neuron_ids)
    neu_biases = generate_bias(n_neu, min_rate, max_rate)
    rate_biased = np.repeat(neu_biases,8)
    rate_biased = r_[[rate_biased]*nsteps]
    x_values = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)
    stim_fixed = r_[[np.linspace(min_rate_fixed,max_rate_fixed,nsteps)]*n_neu*8]
    rate_fixed = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)
    stim_matrix = stim_fixed+rate_biased.T
    timebins = np.linspace(0, 8000,nsteps)

	#create the stimulus according to the stim matrix, we need to sample our tuning curves
    tf,index = ismember(inputpop.soma.laddr,neuron_ids) #pick the right neuron address
    islearning = inputpop[tf].synapses['learning']      #select the synapse
    spiketrain = islearning.spiketrains_inh_poisson(stim_matrix,timebins) #generate spike train from stim matrix

    out_mon = pyNCS.monitors.SpikeMonitor(mypop.soma)	#switch on the monitoring for the events on mypop
    in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma) #switch on the monitoring for the stimulus events 
    setup.monitors.import_monitors([out_mon, in_mon])

	#send the stimulation to the chip
    x = setup.stimulate(spiketrain, send_reset_event=False, tDuration=max(timebins)) 

    # A matrix 
    n_neu = neuron_ids
	#compute mean firing rate of neurons for every stimulation step
    AN = out_mon.sl.firing_rate(1.*max(timebins)/nsteps) 
    a1, a2 = np.shape(AN)
    if a2 == nsteps:
        A = [a[::] for a in AN[n_neu.astype(int),:]]
    else:    
        A = [a[:-1:] for a in AN[n_neu.astype(int),:]]   
    A[:len(n_neu)/2] = [a[::-1] for a in A[:len(n_neu)/2]] 

    if do_plot == 1:
        figure()
        count = 0
        a1, a2 = np.shape(A)
        for p in A:
            count += 1
            if a2 == nsteps:
                if( count <= len(neuron_ids)/2):
                    plot(rate_fixed,p,'mo-') #add color b
                else:
                    plot(rate_fixed,p,'co-') #add color r
            else:
                if( count <= len(neuron_ids)/2):
                    plot(rate_fixed[:-1:],p,'mo-') #add color b
                else:
                    plot(rate_fixed[:-1:],p,'co-') #add color r
        xlabel(r'$\nu_{in} [Hz]$',fontsize=18)
        ylabel(r'$\nu_{out} [Hz]$',fontsize=18)
        string_title = ('Tuning curves for population ' +  pop_name)
        title(string_title)
        show()

    #x between 0 and 1 correspond for us in a stimulation of min_firing_rate,max_firing_rate 
    x_values=[i*2.0/nsteps - 1 for i in range(nsteps)]
    value=np.array([[function(x)] for x in x_values])
    value=np.reshape(value,nsteps)                

    return x_values, A, neu_biases 

def encode_function(setup,chip,mypop,inputpop):

    print 'REMINDER: is the synaptic matrix programmed to be all ones?'

    #setting 
    setMatrixLearning = 0

    min_rate = 0
    max_rate = 45
    min_rate_fixed = 0
    max_rate_fixed = 200
    nsteps = 15
    n_neu = 58
    #noise_level = 50
    
    chip.save_parameters('tmp/biasesOrig.txt')
    chip.load_parameters('biases/new_withQN.txt')

    chip.set_parameter('dpitau',1.725)    
    chip.set_parameter('dpithr',1.63)
    chip.set_parameter('whi',0.46)
    chip.set_parameter('iftau',0.185) 
    chip.set_parameter('ifthr',0.33) #0.285 0.3
    chip.set_parameter('slcaw',1.8)
    chip.set_parameter('slcatau',1.4)
    chip.set_parameter('slthmin',1.8)

    neu_biases = min_rate + (max_rate - min_rate) * random(n_neu)
    rate_biased = np.repeat(neu_biases,8)
    rate_biased = r_[[rate_biased]*nsteps]
    x_values = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)
    stim_fixed = r_[[np.linspace(min_rate_fixed,max_rate_fixed,nsteps)]*n_neu*8]
    rate_fixed = np.linspace(min_rate_fixed,max_rate_fixed,nsteps)
    stim_matrix = stim_fixed+rate_biased.T
    timebins = np.linspace(0, 8000,nsteps)

    islearning = inputpop[:n_neu].synapses['learning']
    spiketrain = islearning.spiketrains_inh_poisson(stim_matrix,timebins)
    #imshow(spiketrain[0].firing_rate(50),aspect='auto',interpolation='nearest',cmap=cm.jet,origin='lower')

    if setMatrixLearning == 1:
        matrixone = np.ones([n_neu,8])
        setSynMatrix(matrixone)

    out_mon = pyNCS.monitors.SpikeMonitor(mypop.soma)
    in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)  
    setup.monitors.import_monitors([out_mon, in_mon])

    x = setup.stimulate(spiketrain, send_reset_event=False, tDuration=max(timebins))

    # A matrix 
    AN = out_mon.sl.firing_rate(1.*max(timebins)/nsteps)
    a1, a2 = np.shape(AN)
    if a2 == nsteps:
        A = [a[::] for a in AN[:n_neu]]
    else:    
        A = [a[:-1:] for a in AN[:n_neu]]   
    A[:n_neu/2] = [a[::-1] for a in A[:n_neu/2]] #flippo A uno ogni 2 non so come ma fabio lo sa, ho capito

    ion()
    [plot(rate_fixed,p,'o-') for p in A]
    xlabel(r'$\nu_{in} [Hz]$',fontsize=18)
    ylabel(r'$\nu_{out} [Hz]$',fontsize=18)

    #function to be encoded  
    x_values=[i*2.0/nsteps - 1 for i in range(nsteps)]
    value=np.array([[function(x)] for x in x_values]) 
    value=np.reshape(value,nsteps)
    #value = np.reshape(value,[1,15])

    # find the optimum linear decoder
    A=np.array(A)#.T
    gamma=np.dot(A, A.T) #I diagonal noise add I = eye(len(gamma))
    upsilon=np.dot(A, value) 
    ginv=np.linalg.pinv(gamma)        
    decoder=np.dot(ginv,upsilon) # i do not have dt in the denumerator ?

    # now that I have decoders 
    x_estimates=np.dot(decoder, [A])

    figure()
    plot(x_values,x_estimates[0],'o-')

	#estimates in new points
    #stim_fixed = r_[[np.linspace(min_rate_fixed,max_rate_fixed+10,nsteps)]*n_neu*8]
    #stim_matrix = stim_fixed+rate_biased.T
    #timebins = np.linspace(0, 8000,nsteps)
    #islearning = inputpop[:n_neu].synapses['learning']
    #spiketrain = islearning.spiketrains_inh_poisson(stim_matrix,timebins)

    x = setup.stimulate(spiketrain, send_reset_event=False, tDuration=max(timebins))

    BN = out_mon.sl.firing_rate(1.*max(timebins)/nsteps)
    b1, b2 = np.shape(BN)
    if b2 == nsteps:
        B = [b[::] for b in BN[:n_neu]]
    else:    
        B = [b[:-1:] for b in BN[:n_neu]]  
    B[:n_neu/2] = [a[::-1] for a in B[:n_neu/2]] #flip A half and half

    x_estimate_chip = np.dot(decoder, [B])
    hold(True)
    plot(x_values,x_estimate_chip[0],'o-')
    plot(x_values,x_estimates[0],'o-')
    xlabel(r'$\vec{x}$',fontsize=18)
    ylabel(r'$f(\vec{x})$',fontsize=18)

    chip.load_parameters('tmp/biasesOrig.txt')

    show()
    return decoder 

def function(x):
    #return cos(x)*sin(x)*2
    #return np.power(x,3)
	return np.power(x,3)

def measure_eff_exc_sram(neuron_ids_b, neuron_ids_a,inputpop,mypop,setup,chip, freqStim=100):
	''' neuron_ids_b (pre)
		neuron_ids_a (post)
    this function measure synaptic efficacy
	it returns efficacy of connection neuron_ids_b->neuron_ids_a '''    
	
	import pyNCS
	import numpy as np

	#chip.set_parameter('iftau',0.16)
	n_stim_step = 2
	duration = np.array([2000.0, 3000.0])
    
	#pre to post
	nsteps = 25
	timebins = np.linspace(0, duration[0], nsteps)

    #back to att
	efficacy_matrix_back_to_att = np.zeros([len(neuron_ids_b), len(neuron_ids_a)])
	for neu in range(len(neuron_ids_b)):
		stim_fixed = r_[[np.linspace(freqStim,freqStim,n_stim_step)]*7] #stim comes in in all 7 syn
		sram_e = inputpop[neuron_ids_b[neu]].synapses['sram_exc']
		spiketrain_e = sram_e.spiketrains_inh_poisson(stim_fixed,duration)

		#monitoring
		out_mon = pyNCS.monitors.SpikeMonitor(mypop.soma)
		in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)
		setup.monitors.import_monitors([out_mon, in_mon])

		x = setup.stimulate(spiketrain_e, send_reset_event=False, duration=max(timebins))
		firing_rates = out_mon.sl.firing_rate(np.max(duration))
		firing_rates = firing_rates[:,0]
		this_pre_freq = firing_rates[neuron_ids_b[neu]]
		raw_data = out_mon.sl.raw_data()
		nspike_pre = np.size(np.where(raw_data[:,1] == neuron_ids_b[neu]))
		print 'spike pre', nspike_pre
		for this_neu_post in range(len(neuron_ids_a)):
			nspike_post = np.size(np.where(raw_data[:,1] == neuron_ids_a[this_neu_post]))
			print 'spike post', nspike_post
			print 'neu', neu
			print 'this_neu_post', neuron_ids_a[this_neu_post]
			if(nspike_post > 0):
				efficacy = np.float(nspike_post)/np.float(nspike_pre)
			else:
				efficacy = 0
			#if(neu != neuron_ids_a[this_neu_post]):
			efficacy_matrix_back_to_att[neu][this_neu_post] = efficacy
 

	return efficacy_matrix_back_to_att

def measure_eff_inh_sram(neuron_ids_b, neuron_ids_a,inputpop,mypop,setup,chip, freqStim=100, inj=1.5):
	''' neuron_ids_b (pre)
		neuron_ids_a (post)
    this function measure synaptic efficacy
	it returns efficacy of connection neuron_ids_b->neuron_ids_a '''    
	
	import pyNCS
	import numpy as np

	#chip.set_parameter('iftau',0.16)
	n_stim_step = 2
	duration = np.array([2000.0, 3000.0])
    
	#pre to post
	nsteps = 25
	timebins = np.linspace(0, duration[0], nsteps)

	inj_0 = chip.get_parameter('ifdc')
	chip.set_parameter('ifdc',inj)

    #back to att
	efficacy_matrix_back_to_att = np.zeros([len(neuron_ids_b), len(neuron_ids_a)])

	#make the zero measurements
	out_mon = pyNCS.monitors.SpikeMonitor(mypop.soma)
	in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)
	setup.monitors.import_monitors([out_mon, in_mon])

	x = setup.stimulate({}, send_reset_event=False, duration=max(timebins))

	for neu in range(len(neuron_ids_b)):
		stim_fixed = r_[[np.linspace(freqStim,freqStim,n_stim_step)]*7] #stim comes in in all 7 syn
		sram_e = inputpop[neuron_ids_b[neu]].synapses['sram_exc']
		spiketrain_e = sram_e.spiketrains_inh_poisson(stim_fixed,duration)

		#monitoring
		out_mon = pyNCS.monitors.SpikeMonitor(mypop.soma)
		in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)
		setup.monitors.import_monitors([out_mon, in_mon])

		x = setup.stimulate(spiketrain_e, send_reset_event=False, duration=max(timebins))
		firing_rates = out_mon.sl.firing_rate(np.max(duration))
		firing_rates = firing_rates[:,0]
		this_pre_freq = firing_rates[neuron_ids_b[neu]]
		raw_data = out_mon.sl.raw_data()
		nspike_pre = np.size(np.where(raw_data[:,1] == neuron_ids_b[neu]))
		print 'spike pre', nspike_pre
		for this_neu_post in range(len(neuron_ids_a)):
			nspike_post = np.size(np.where(raw_data[:,1] == neuron_ids_a[this_neu_post]))
			print 'spike post', nspike_post
			print 'neu', neu
			print 'this_neu_post', neuron_ids_a[this_neu_post]
			if(nspike_post > 0):
				if(nspike_pre > 0):
					efficacy = np.float(nspike_post)/np.float(nspike_pre)
				else:
					print 'WARNING pre neuron never fired, neu: ', neuron_ids_b[neu] 
					efficacy = 1
			else:
				efficacy = 0
			#if(neu != neuron_ids_a[this_neu_post]):
			efficacy_matrix_back_to_att[neu][this_neu_post] = efficacy
 
	chip.set_parameter('ifdc',inj_0)

	return efficacy_matrix_back_to_att

def root_mean_square(ideal, measured):
    ''' calculate RMSE 
    ie: root_mean_square(ideal,measured)
    numpy vector in 
    float out'''

    return np.sqrt(((ideal - measured) ** 2).mean())


#calibrate sram
def calibrate_sram_w(chip, mypop, inputpop, setup, w_min=0, w_max=15, rate_stimulation = 500, stim_dur = 1000):
    ''' this function is used to measure the transfer characteristic of neurons 
    and save them in a file. This is part of the miss-match compensation procedure 
    es: matrix_cal = nef.calibrate_sram_w(chip, mypop, inputpop,setup,rate_stimulation=600)
    chip.load_parameters('biases/sram_calib_ok.txt')
    '''

    #chip.load_parameters('biases/sram_calib_ok.txt')

    n_neu = 58 # all the array
    digitalw = np.linspace(w_min,w_max,w_max-w_min+1)
    matrix_calibration = [] #[neu_id syneff digitalw]
    for this_digital_w in range(int(max(digitalw))+1):
        #stimulate with a regular spike train all neurons and measure syn eff
    
        #monitor on
        out_mon = pyNCS.monitors.SpikeMonitor(mypop.soma)
        in_mon = pyNCS.monitors.SpikeMonitor(inputpop.soma)  
        setup.monitors.import_monitors([out_mon, in_mon])
    
        this_syn = inputpop[:].synapses['sram_exc'].addr['syn'] == 1
        write_on = inputpop[:].synapses['sram_exc'].addr['write'] == 1
        this_w = inputpop[:].synapses['sram_exc'].addr['weight'] == this_digital_w
        tf_index = this_syn & this_w & write_on
        index_int = np.where(tf_index)
        issram = inputpop[:].synapses['sram_exc'][index_int]
        spiketrain = issram.spiketrains_regular(rate_stimulation,t_start=0,duration=stim_dur)
        
        x = setup.stimulate(spiketrain, send_reset_event=False)

        sentSpikesToThisNeu = len(spiketrain[10].raw_data())/58
        raw_output_data = out_mon.sl.raw_data()

        for this_neu in range(58):
            spike_out_this_neu = len(np.where(raw_output_data[:,1] == this_neu)[0])
            synapticEfficacy =  (spike_out_this_neu/float(sentSpikesToThisNeu))
            matrix_calibration.append([this_neu,this_digital_w,synapticEfficacy])
        

    matrix_calibration = np.array(matrix_calibration)
    np.savetxt('matrix_calibration.txt', matrix_calibration)

    return matrix_calibration

def plot_calibration_sram(matrix_cal):
    ''' this function plot the calibration matrix '''

    ion()
    figure()
    hold(True)
    for neu in range(58):
        index_this_neu = np.where(matrix_cal[:,0] == neu)   
        plot(matrix_cal[index_this_neu,1][0],matrix_cal[index_this_neu,2][0],'o-')   
    xlabel(r'digital w', fontsize=18)
    ylabel(r'J( $\theta$ - H)', fontsize=18)
    show()
    return

def compute_decoders_convex_bounded(x_values,A,nsteps,function=lambda x:x, min_x = -0.2, max_x = 0.2):
    '''solve convex optimization problem with cvxpy '''
    import cvxpy as cx
    import numpy as np

    A_m = np.matrix(np.matrix(A))
    aa = cx.zeros(np.shape(A))
    aa[A_m.nonzero()] = A_m[A_m.nonzero()]
    
    #function to be encoded  
    value=np.array([[function(x)] for x in x_values]) 
    value=np.reshape(value,nsteps)

    bb = cx.zeros(np.shape(value))
    bb[0,value.nonzero()[0]] = value[value.nonzero()[0]]    

    m,n = np.shape(aa)
    dec = cx.variable(m)
    
    p = cx.program(cx.minimize(cx.norm2(np.transpose(aa)*(dec)-np.transpose(bb))), [cx.leq(dec,max_x),cx.geq(dec,min_x)])
    p.solve()

    return dec.value


def get_digital_efficacy(neuid, desired_w, matrix_cal):
    ''' this function read the calibration file and return the digital 
    sram weight to match for a desired synaptic efficacy
    es: nef.get_digital_efficacy(5,0.2,matrix_cal)
    '''

    index_this_neu = np.where(matrix_cal[:,0] == neuid) 
    X = matrix_cal[index_this_neu,1][0]
    Y = matrix_cal[index_this_neu,2][0]
    
    from scipy.interpolate import UnivariateSpline
    extrapolator = UnivariateSpline( Y, X, k=1)

    y = extrapolator( desired_w )
    digital_w = ceil(y)

    if(digital_w < 0):
        print 'warning digital w smaller than min value (0)'
        digital_w = 0
    if(digital_w > 15):
        print 'warning digital w greater than max value (15)'
        digital_w = 15

    return digital_w

def non_calibrate_sram_w(w_min=0,w_max=15):
    ''' return uncalibrated weight, straight line '''
    matrix_cal = []
    digitalw = np.linspace(w_min,w_max,w_max-w_min+1)
    normw = np.linspace(0,1,16)
    for i in range(58):
        for this_digital_w in range(len(digitalw)):
            matrix_cal.append([i,digitalw[this_digital_w],normw[this_digital_w]])

    matrix_cal = np.array(matrix_cal)
    return matrix_cal
