import numpy as np
from pylab import *
import matplotlib
import os as os
import pyNCS
from time import time

##### Mean firing rate of each neuron
def meanNeuFiring(SpikeTrain, n_neurons,simulation_time):
	import numpy as np
	ArraySpike = np.array(SpikeTrain)
	MeanRate   = np.zeros([len(n_neurons)])
	for i in range(len(n_neurons)):
		Index_n = np.where(np.logical_and(ArraySpike[:,1] == n_neurons[i], np.logical_and(ArraySpike[:,0] > simulation_time[0] , ArraySpike[:,0] < simulation_time[1] )) )
		MeanRate[i] = len(Index_n[0])*1000.0/(simulation_time[1]-simulation_time[0]) # time unit: ms
	return MeanRate

def measurePhy_all(d, setup,chip,mypop,inputpop,out_mon):

    import time as time
    import os as os
    import getopt
    import pyNCS
    et=pyNCS.et
    d=et.mksavedir('results')

    #lets make a network of 6 neurons
    #acceleration excitation - deceleration inhibition 
    #regular afferent
    regular_x = np.array([10])
    regular_y = np.array([29]) #21 23
    regular_z = np.array([19])

    #irregular afferent
    irregular_x = np.array([28]) #13
    irregular_y = np.array([6])
    irregular_z = np.array([36])

    #load biases
    #chip.load_parameters('biases/vestibular_network.txt')


    chip.load_parameters('biases/new_vestibular_adapt.txt')
    #chip.load_parameters('biases/cv_irregular_neuron.txt')

    chip.set_parameter('ifdc',1.8)#71
    chip.set_parameter('iftau',0.17)
    chip.set_parameter('ifthr',0.375) #66
    chip.set_parameter('ifrfr',0.3)

    chip.set_parameter('dpitau',1.628)
    chip.set_parameter('dpithr',1.478) #irregular if 1.49 --> regular 1.46

    chip.set_parameter('whi',0.5)
    chip.set_parameter('wthr',0.28) #0.28
    
    chip.set_parameter('ahtau',0)
    chip.set_parameter('ahw',1.2)   #no adapt 1.8
    chip.set_parameter('athr',0.08)  #no adapt 0 
    chip.set_parameter('ifcasc',0.3) # no adapt 0

    chip.set_parameter('ifdc',1.8)    
    chip.set_parameter('pwlk',1.5) 

    #sram bias
    chip.set_parameter('SYN_RAM_DAC1', 0.43)
    chip.set_parameter('SYN_RAM_DAC0', 0.43)
    chip.set_parameter('SYN_RAM_DAC2', 0.43)
    chip.set_parameter('SYN_RAM_DAC3', 0.43)

    chip.set_parameter('dpithr_sram',1.3)
    chip.set_parameter('dpitau_sram_exc',1.67)
    chip.set_parameter('dpitau_sram_inh',1.62)
    chip.set_parameter('SYNPU',1.2)

    figure_counter = int(time.time()) 
    figure_directory = d + 'figures/'
    if os.path.isdir(figure_directory):
        print 'figure directory already exists..'
    else:
        os.mkdir(figure_directory)

    ##################################
    # CALIBRATION ISI HISTOGRAMS
    ##################################

    #setup the stimulation protocol
    stepSize = 8000
    minFreq = 1000
    maxFreq = 1000
    nstepf = 20
    frequencies = np.linspace(minFreq,maxFreq,nstepf)
    durations =  np.arange(0,8000,len(frequencies))#np.arange(0,20000,1000)  np.arange(0,7500,500)#    
    #[1,10,20,42,65,80,90,108,110,120,130,140,150,160,170,180,200,220,250,280];#  [50,55,60,65,70]#   

    #set up the sigma bias for irregular neurons
    spikeTrain_x_i = inputpop[int(irregular_x)].synapses['sram_inh'][0].spiketrains_regular_gaussian(150,scale=50,t_start=0,duration=stepSize)
    spikeTrain_x_e = inputpop[int(irregular_x)].synapses['sram_exc'][1].spiketrains_poisson(0.1,t_start=0,duration=stepSize)
    spikeTrain_y_i = inputpop[int(irregular_y)].synapses['sram_inh'][0].spiketrains_regular_gaussian(150,scale=50,t_start=0,duration=stepSize)
    spikeTrain_y_e = inputpop[int(irregular_x)].synapses['sram_exc'][1].spiketrains_poisson(0.1,t_start=0,duration=stepSize)
    spikeTrain_z_i = inputpop[int(irregular_z)].synapses['sram_inh'][0].spiketrains_regular_gaussian(150,scale=50,t_start=0,duration=stepSize)
    spikeTrain_z_e = inputpop[int(irregular_z)].synapses['sram_exc'][1].spiketrains_poisson(0.1,t_start=0,duration=stepSize)

    #merge spikelists irregular neurons x,y,z 
    tot_spike_noise = pyNCS.pyST.merge_spikelists(spikeTrain_x_i[10],spikeTrain_y_i[10],spikeTrain_z_i[10],spikeTrain_x_e[10],spikeTrain_y_e[10],spikeTrain_z_e[10])
    tot_spike_noise = {10:tot_spike_noise}
   
    #monitors
    out_mon_reg_x = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_x)])
    out_mon_reg_y = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_y)])
    out_mon_reg_z = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_z)])
    out_mon_irreg_x = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_x)])
    out_mon_irreg_y = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_y)])
    out_mon_irreg_z = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_z)])

    setup.monitors.import_monitors([out_mon_reg_x,out_mon_reg_y,out_mon_reg_z,out_mon_irreg_x,out_mon_irreg_y,out_mon_irreg_z])


    #stimulate the chip and record spiketrain in the monitors
    x = setup.stimulate(tot_spike_noise, send_reset_event=False, tDuration=max(durations))
	
    #make regular and irregular histograms
    spk_reg_x = out_mon_reg_x.sl.raw_data()
    spk_reg_y = out_mon_reg_y.sl.raw_data()
    spk_reg_z = out_mon_reg_z.sl.raw_data()

    spk_irreg_x = out_mon_irreg_x.sl.raw_data()
    spk_irreg_y = out_mon_irreg_y.sl.raw_data()
    spk_irreg_z = out_mon_irreg_z.sl.raw_data()

    np.savetxt(d+'spk_reg_z.txt',spk_reg_z)
    np.savetxt(d+'spk_irreg_z.txt',spk_irreg_z)    
    np.savetxt(d+'spk_reg_x.txt',spk_reg_x)
    np.savetxt(d+'spk_irreg_x.txt',spk_irreg_x)
    np.savetxt(d+'spk_reg_y.txt',spk_reg_y)
    np.savetxt(d+'spk_irreg_y.txt',spk_irreg_y)

    #plot histogram
    figure()
    isi_irreg_z = diff(spk_irreg_z[:,0]*1e-3)
    hist(isi_irreg_z,15,normed=True)
    xlabel('ISI [s]',fontsize=20)
    ylabel('counts')   
    filename = str(figure_directory)+str(figure_counter)+str('_hist_regular.png')
    savefig(filename)     
    figure_counter += 1

    figure()
    isi_reg_z = diff(spk_reg_z[:,0]*1e-3)
    hist(isi_reg_z,15,normed=True)
    xlabel('ISI [s]',fontsize=20)   
    ylabel('counts')   
    filename = str(figure_directory)+str(figure_counter)+str('_hist_irregular.png')
    savefig(filename)     
    figure_counter += 1

    ##################################
    # CALIBRATION F F CURVE
    ##################################
    #lets make the f f curve for all the different neurons   
    #setup the stimulation protocol
    stepSize = 500
    durations =  np.arange(0,20000,500)#np.arange(0,20000,1000)  np.arange(0,7500,500)#    
    minFreq = 250
    maxFreq = 1400
    nstepf = 20
    frequencies = np.linspace(minFreq,maxFreq,nstepf)
    numtrials = 3
    #[1,10,20,42,65,80,90,108,110,120,130,140,150,160,170,180,200,220,250,280];#  [50,55,60,65,70]#   

    tot_neuron_activity_irrex_x = []
    tot_neuron_activity_irrex_y = []
    tot_neuron_activity_irrex_z = []
    tot_neuron_activity_rex_x = []
    tot_neuron_activity_rex_y = []
    tot_neuron_activity_rex_z = []
    for i in range(numtrials):
        spikeTrainTotal = []
        #set up the train stimulus learn syn
        for i in range(len(frequencies)):
            #regular contribution
            spikeTrain_f_reg_x = inputpop[int(regular_x)].synapses['sram_exc'][2].spiketrains_regular(frequencies[i],t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_f_reg_x)
            spikeTrain_f_irreg_x = inputpop[int(irregular_x)].synapses['sram_exc'][3].spiketrains_regular(frequencies[i],t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_f_irreg_x)
            spikeTrain_f_reg_y = inputpop[int(regular_y)].synapses['sram_exc'][2].spiketrains_regular(frequencies[i]+650*(1+i),t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_f_reg_y)
            spikeTrain_f_irreg_y = inputpop[int(irregular_y)].synapses['sram_exc'][3].spiketrains_regular(frequencies[i],t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_f_irreg_y)
            spikeTrain_f_reg_z = inputpop[int(regular_z)].synapses['sram_exc'][2].spiketrains_regular(frequencies[i],t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_f_reg_z)
            spikeTrain_f_irreg_z = inputpop[int(irregular_z)].synapses['sram_exc'][3].spiketrains_regular(frequencies[i],t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_f_irreg_z)
            #noise contribution for irregular neurons
            spikeTrain_x_i = inputpop[int(irregular_x)].synapses['sram_inh'][0].spiketrains_poisson(180,t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_x_i)
            spikeTrain_x_e = inputpop[int(irregular_x)].synapses['sram_exc'][1].spiketrains_poisson(85,t_start=0,duration=stepSize)
            spikeTrainTotal.append(spikeTrain_x_e)
            spikeTrain_y_i = inputpop[int(irregular_y)].synapses['sram_inh'][0].spiketrains_poisson(150,t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_y_i)
            spikeTrain_y_e = inputpop[int(irregular_y)].synapses['sram_exc'][1].spiketrains_poisson(85,t_start=0,duration=stepSize)
            spikeTrainTotal.append(spikeTrain_y_e)
            spikeTrain_z_i = inputpop[int(irregular_z)].synapses['sram_inh'][0].spiketrains_poisson(55,t_start=durations[i],duration=stepSize)
            spikeTrainTotal.append(spikeTrain_z_i)
            spikeTrain_z_e = inputpop[int(irregular_z)].synapses['sram_exc'][1].spiketrains_poisson(200,t_start=durations[i],duration=stepSize)      
            spikeTrainTotal.append(spikeTrain_z_e)  


        #merge spikelists  
        totspkLearn = pyNCS.pyST.merge_sequencers(*spikeTrainTotal)   

        #monitors
        out_mon_reg_x = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_x)])
        out_mon_reg_y = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_y)])
        out_mon_reg_z = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_z)])
        out_mon_irreg_x = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_x)])
        out_mon_irreg_y = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_y)])
        out_mon_irreg_z = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_z)])

        setup.monitors.import_monitors([out_mon_reg_x,out_mon_reg_y,out_mon_reg_z,out_mon_irreg_y,out_mon_irreg_z,out_mon_irreg_x])

        #stimulate
        x = setup.stimulate(totspkLearn, send_reset_event=False, tDuration=max(durations))

        neuron_activity_irrex_x = compute_neuron_activity(out_mon_irreg_x.sl, frequencies)
        neuron_activity_irrex_y = compute_neuron_activity(out_mon_irreg_y.sl, frequencies)
        neuron_activity_irrex_z = compute_neuron_activity(out_mon_irreg_z.sl, frequencies)
        neuron_activity_rex_z = compute_neuron_activity(out_mon_reg_z.sl, frequencies)
        neuron_activity_rex_x = compute_neuron_activity(out_mon_reg_x.sl, frequencies)
        neuron_activity_rex_y = compute_neuron_activity(out_mon_reg_y.sl, frequencies)

        tot_neuron_activity_irrex_x.append(neuron_activity_irrex_x)
        tot_neuron_activity_irrex_y.append(neuron_activity_irrex_y)
        tot_neuron_activity_irrex_z.append(neuron_activity_irrex_z)
        tot_neuron_activity_rex_x.append(neuron_activity_rex_x)
        tot_neuron_activity_rex_y.append(neuron_activity_rex_y)
        tot_neuron_activity_rex_z.append(neuron_activity_rex_z)



    npoints = len(tot_neuron_activity_rex_z[0])

    tot_neuron_activity_irrex_x = np.array(tot_neuron_activity_irrex_x)
    tot_neuron_activity_irrex_y = np.array(tot_neuron_activity_irrex_y)
    tot_neuron_activity_irrex_z = np.array(tot_neuron_activity_irrex_z)
    tot_neuron_activity_rex_x = np.array(tot_neuron_activity_rex_x)
    tot_neuron_activity_rex_y = np.array(tot_neuron_activity_rex_y)
    tot_neuron_activity_rex_z = np.array(tot_neuron_activity_rex_z)

    std_tot_neuron_activity_irrex_x = np.zeros(npoints)
    std_tot_neuron_activity_irrex_y = np.zeros(npoints)
    std_tot_neuron_activity_irrex_z = np.zeros(npoints)
    std_tot_neuron_activity_rex_x = np.zeros(npoints)
    std_tot_neuron_activity_rex_y = np.zeros(npoints)
    std_tot_neuron_activity_rex_z = np.zeros(npoints)

    mean_tot_neuron_activity_irrex_x = np.zeros(npoints)
    mean_tot_neuron_activity_irrex_y = np.zeros(npoints)
    mean_tot_neuron_activity_irrex_z = np.zeros(npoints)
    mean_tot_neuron_activity_rex_x = np.zeros(npoints)
    mean_tot_neuron_activity_rex_y = np.zeros(npoints)
    mean_tot_neuron_activity_rex_z = np.zeros(npoints)

    for this in range(npoints):
        std_tot_neuron_activity_irrex_x[this] = np.std(tot_neuron_activity_irrex_x[:,this])
        std_tot_neuron_activity_irrex_y[this] = np.std(tot_neuron_activity_irrex_y[:,this])
        std_tot_neuron_activity_irrex_z[this] = np.std(tot_neuron_activity_irrex_z[:,this])
        std_tot_neuron_activity_rex_x[this] = np.std(tot_neuron_activity_rex_x[:,this])
        std_tot_neuron_activity_rex_y[this] = np.std(tot_neuron_activity_rex_y[:,this])
        std_tot_neuron_activity_rex_z[this] = np.std(tot_neuron_activity_rex_z[:,this])
        mean_tot_neuron_activity_irrex_x[this] = np.mean(tot_neuron_activity_irrex_x[:,this])
        mean_tot_neuron_activity_irrex_y[this] = np.mean(tot_neuron_activity_irrex_y[:,this])
        mean_tot_neuron_activity_irrex_z[this] = np.mean(tot_neuron_activity_irrex_z[:,this])
        mean_tot_neuron_activity_rex_x[this] = np.mean(tot_neuron_activity_rex_x[:,this])
        mean_tot_neuron_activity_rex_y[this] = np.mean(tot_neuron_activity_rex_y[:,this])
        mean_tot_neuron_activity_rex_z[this] = np.mean(tot_neuron_activity_rex_z[:,this])

    #plot the transfer function
    figure()
    plot(frequencies[1:-1:],mean_tot_neuron_activity_irrex_x[1:-1:],'bo-', label='irregular')
    fill_between(frequencies[1:-1:], mean_tot_neuron_activity_irrex_x[1:-1:]-std_tot_neuron_activity_irrex_x[1:-1:], mean_tot_neuron_activity_irrex_x[1:-1:]+std_tot_neuron_activity_irrex_x[1:-1:],alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=2, antialiased=True)
    hold(True)
    plot(frequencies[1:-1:],mean_tot_neuron_activity_irrex_y[1:-1:],'bo-')
    fill_between(frequencies[1:-1:], mean_tot_neuron_activity_irrex_y[1:-1:]-std_tot_neuron_activity_irrex_y[1:-1:], mean_tot_neuron_activity_irrex_y[1:-1:]+std_tot_neuron_activity_irrex_y[1:-1:],alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=2, antialiased=True)
    hold(True)
    plot(frequencies[1:-1:],mean_tot_neuron_activity_irrex_z[1:-1:],'bo-')
    fill_between(frequencies[1:-1:], mean_tot_neuron_activity_irrex_z[1:-1:]-std_tot_neuron_activity_irrex_z[1:-1:], mean_tot_neuron_activity_irrex_z[1:-1:]+std_tot_neuron_activity_irrex_z[1:-1:],alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=2, antialiased=True)
    hold(True)
    plot(frequencies[1:-1:],mean_tot_neuron_activity_rex_z[1:-1:],'o-', color='#CC4F1B',  label='regular')
    fill_between(frequencies[1:-1:], mean_tot_neuron_activity_rex_z[1:-1:]-std_tot_neuron_activity_rex_z[1:-1:], mean_tot_neuron_activity_rex_z[1:-1:]+std_tot_neuron_activity_rex_z[1:-1:],alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848',linewidth=2, antialiased=True)    
    hold(True)
    plot(frequencies[1:-1:],mean_tot_neuron_activity_rex_y[1:-1:],'o--',color='#CC4F1B')
    fill_between(frequencies[1:-1:], mean_tot_neuron_activity_rex_y[1:-1:]-std_tot_neuron_activity_rex_y[1:-1:], mean_tot_neuron_activity_rex_y[1:-1:]+std_tot_neuron_activity_rex_y[1:-1:],alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848',linewidth=2, antialiased=True)
    hold(True)
    plot(frequencies[1:-1:],mean_tot_neuron_activity_rex_x[1:-1:],'o-',color='#CC4F1B')
    fill_between(frequencies[1:-1:], mean_tot_neuron_activity_rex_x[1:-1:]-std_tot_neuron_activity_rex_x[1:-1:], mean_tot_neuron_activity_rex_x[1:-1:]+std_tot_neuron_activity_rex_x[1:-1:],alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848',linewidth=2, antialiased=True)
    grid(True)
    xlabel(r'$\nu_{in}$ [Hz]', fontsize=22)
    ylabel(r'$\nu_{out}$ [Hz]', fontsize=22)
    legend(loc='upper left')
    filename = str(figure_directory)+str(figure_counter)+str('_fVsf.png')
    savefig(filename)     
    figure_counter += 1

    et.save(out_mon_reg_x,'out_mon_reg_x_calib.pickle')
    et.save(out_mon_reg_y,'out_mon_reg_y_calib.pickle')
    et.save(out_mon_reg_z,'out_mon_reg_z_calib.pickle')
    et.save(out_mon_irreg_x,'out_mon_irreg_x_calib.pickle')
    et.save(out_mon_irreg_y,'out_mon_irreg_y_calib.pickle')
    et.save(out_mon_irreg_z,'out_mon_irreg_z_calib.pickle')
    np.savetxt(d+'input_frequencies_calib.txt',frequencies)
    et.save(inputpop,'inputpop_calib.pickle')
    chip.save_parameters(d+'biases_chip')


    ##################################
    # EXPERIMENT WITH SENSOR DATA
    ##################################
    ins = open("sensor_data/acquisitionFreqConversion13_6_2013.txt", "r" )
    array = []
    count = 0
    for line in ins:
        array.append( line )
        count = count+1;

    #extract data into array
    dx = []
    dy = []
    dz = []
    tmp_a = 0.0
    tmp_b = 0.0
    tmp_c = 0.0
    dms = np.zeros(count)
    dns = np.zeros(count)
    for index, item in enumerate(array):
        tmp_a,tmp_b,tmp_c = item.rsplit(',')
        tmp_a = tmp_a.replace(',','')
        tmp_b = tmp_b.replace(',','')
        if(int(tmp_c) == 0):
            dx.append(float(tmp_b))
        if(int(tmp_c) == 1):
            dy.append(float(tmp_b))     
        if(int(tmp_c) == 2):
            dz.append(float(tmp_b))     

    dx = np.array(dx)
    dy = np.array(dy)
    dz = np.array(dz)
  
    index_x = np.nonzero(diff(dx)<0)
    index_y = np.nonzero(diff(dy)<0)
    index_z = np.nonzero(diff(dz)<0)

    #dx[index_x[0][0]+1::]+dx[index_x[0][0]]
    resetted_times = len(index_x[0])
    for i in range(resetted_times):
        dx[index_x[0][i]+1::] = dx[index_x[0][i]+1::]+1000000
        dy[index_y[0][i]+1::] = dy[index_y[0][i]+1::]+1000000
        dz[index_z[0][i]+1::] = dz[index_z[0][i]+1::]+1000000


    figure()
    plot(diff(dx),'*', label='x input')
    xlabel('time [ms]', fontsize=22)
    ylabel('au', fontsize=22)
    legend(loc='upper right')    
    filename = str(figure_directory)+str(figure_counter)+str('_input_sensor_data_x.png')
    savefig(filename)     
    figure_counter += 1

    figure()
    plot(diff(dy),'*', label='y input')
    xlabel('time [ms]', fontsize=22)
    ylabel('au', fontsize=22)
    legend(loc='upper right')
    filename = str(figure_directory)+str(figure_counter)+str('_input_sensor_data_y.png')
    savefig(filename)     
    figure_counter += 1

    figure()
    plot(diff(dz),'*', label='z input')
    xlabel('time [ms]', fontsize=22)
    ylabel('au', fontsize=22)
    legend(loc='upper right')
    filename = str(figure_directory)+str(figure_counter)+str('_input_sensor_data_z.png')
    savefig(filename)     
    figure_counter += 1

    #make the spike train with those data
    raw_stim_reg_x = np.array([repeat(regular_x,len(dx)),dx]).T
    raw_stim_reg_y = np.array([dy,repeat(regular_y,len(dy))]).T
    raw_stim_reg_z = np.array([dz,repeat(regular_z,len(dz))]).T
    raw_stim_irreg_x = np.array([dx,repeat(irregular_x,len(dx))]).T
    raw_stim_irreg_y = np.array([dy,repeat(irregular_y,len(dy))]).T
    raw_stim_irreg_z = np.array([dz,repeat(irregular_z,len(dz))]).T

    #make the spike train with those data
    ## REGULAR NEURONS  
    this_syn_address = inputpop[int(regular_x)].synapses['sram_exc'][2].laddr
    raw_stim_reg_x = np.array([repeat(this_syn_address,len(dx)),(dx-min(dx)+1)*10e-4]).T
    raw_stim_reg_x_tuple = tuple(tuple(x) for x in raw_stim_reg_x)
    spiketrain_reg_x = inputpop[int(regular_x)].synapses['sram_exc'][2].spiketrains(raw_stim_reg_x_tuple)

    this_syn_address = inputpop[int(regular_y)].synapses['sram_exc'][2].laddr
    raw_stim_reg_y = np.array([repeat(this_syn_address,len(dy)),(dy-min(dy)+1)*10e-4]).T
    raw_stim_reg_y_tuple = tuple(tuple(x) for x in raw_stim_reg_y)
    spiketrain_reg_y = inputpop[int(regular_y)].synapses['sram_exc'][2].spiketrains(raw_stim_reg_y_tuple)

    this_syn_address = inputpop[int(regular_z)].synapses['sram_exc'][2].laddr
    raw_stim_reg_z = np.array([repeat(this_syn_address,len(dz)),(dz-min(dz)+1)*10e-4]).T
    raw_stim_reg_z_tuple = tuple(tuple(x) for x in raw_stim_reg_z)
    spiketrain_reg_z = inputpop[int(regular_z)].synapses['sram_exc'][2].spiketrains(raw_stim_reg_z_tuple)
  
    ## IRREGULAR NEURONS we need to add a bit of noise
    this_syn_address = inputpop[int(irregular_x)].synapses['sram_exc'][2].laddr
    raw_stim_irreg_x = np.array([repeat(this_syn_address,len(dx)),(dx-min(dx)+1)*10e-4]).T
    raw_stim_irreg_x_tuple = tuple(tuple(x) for x in raw_stim_irreg_x)
    spiketrain_irreg_x = inputpop[int(irregular_x)].synapses['sram_exc'][2].spiketrains(raw_stim_irreg_x_tuple)

    this_syn_address = inputpop[int(irregular_y)].synapses['sram_exc'][2].laddr
    raw_stim_irreg_y = np.array([repeat(this_syn_address,len(dy)),(dy-min(dy)+1)*10e-4]).T
    raw_stim_irreg_y_tuple = tuple(tuple(x) for x in raw_stim_irreg_y)
    spiketrain_irreg_y = inputpop[int(irregular_y)].synapses['sram_exc'][2].spiketrains(raw_stim_irreg_y_tuple)

    this_syn_address = inputpop[int(irregular_z)].synapses['sram_exc'][2].laddr
    raw_stim_irreg_z = np.array([repeat(this_syn_address,len(dz)),(dz-min(dz)+1)*10e-4]).T
    raw_stim_irreg_z_tuple = tuple(tuple(x) for x in raw_stim_irreg_z)
    spiketrain_irreg_z = inputpop[int(irregular_z)].synapses['sram_exc'][2].spiketrains(raw_stim_irreg_z_tuple)
 
    spikeTrainTotal = []
    sizeStep = max((dz-min(dz)+1)*10e-4)
    #noise contribution for irregular neurons
    spikeTrain_x_i = inputpop[int(irregular_x)].synapses['sram_inh'][0].spiketrains_poisson(180,t_start=0,duration=sizeStep)
    spikeTrainTotal.append(spikeTrain_x_i)
    spikeTrain_x_e = inputpop[int(irregular_x)].synapses['sram_exc'][1].spiketrains_poisson(85,t_start=0,duration=sizeStep)
    spikeTrainTotal.append(spikeTrain_x_e)
    spikeTrain_y_i = inputpop[int(irregular_y)].synapses['sram_inh'][0].spiketrains_poisson(150,t_start=0,duration=sizeStep)
    spikeTrainTotal.append(spikeTrain_y_i)
    spikeTrain_y_e = inputpop[int(irregular_y)].synapses['sram_exc'][1].spiketrains_poisson(85,t_start=0,duration=sizeStep)
    spikeTrainTotal.append(spikeTrain_y_e)
    spikeTrain_z_i = inputpop[int(irregular_z)].synapses['sram_inh'][0].spiketrains_poisson(55,t_start=0,duration=sizeStep)
    spikeTrainTotal.append(spikeTrain_z_i)
    spikeTrain_z_e = inputpop[int(irregular_z)].synapses['sram_exc'][1].spiketrains_poisson(200,t_start=0,duration=sizeStep)      
    spikeTrainTotal.append(spikeTrain_z_e)
    totspk_noise_irreg = pyNCS.pyST.merge_sequencers(*spikeTrainTotal)   

    #merge regular neurons
    tot_spike_regular = pyNCS.pyST.merge_spikelists(spiketrain_reg_x[10],spiketrain_reg_y[10],spiketrain_reg_z[10])
    tot_spike_regular = {10:tot_spike_regular}
    tot_spike_regular[10].raster_plot()
    
    #merge irregular neurons
    tot_spike_irregular = pyNCS.pyST.merge_spikelists(spiketrain_irreg_x[10],spiketrain_irreg_y[10],spiketrain_irreg_z[10],totspk_noise_irreg[10])
    tot_spike_irregular = {10:tot_spike_irregular}
    tot_spike_irregular[10].raster_plot()

    #merge regular and irregular
    tot_spk = pyNCS.pyST.merge_spikelists(tot_spike_irregular[10],tot_spike_regular[10])
    tot_spk = {10:tot_spk}

    #monitors
    out_mon_reg_x = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_x)])
    out_mon_reg_y = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_y)])
    out_mon_reg_z = pyNCS.monitors.SpikeMonitor(mypop.soma[int(regular_z)])
    out_mon_irreg_x = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_x)])
    out_mon_irreg_y = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_y)])
    out_mon_irreg_z = pyNCS.monitors.SpikeMonitor(mypop.soma[int(irregular_z)])

    setup.monitors.import_monitors([out_mon_reg_x,out_mon_reg_y,out_mon_reg_z,out_mon_irreg_y,out_mon_irreg_z,out_mon_irreg_x])

    #stimulate
    x = setup.stimulate(tot_spk, send_reset_event=False, tDuration=max(durations))


    et.save(out_mon_reg_x,'out_mon_reg_x_exp.pickle')
    et.save(out_mon_reg_y,'out_mon_reg_y_exp.pickle')
    et.save(out_mon_reg_z,'out_mon_reg_z_exp.pickle')
    et.save(out_mon_irreg_x,'out_mon_reg_x_exp.pickle')
    et.save(out_mon_irreg_y,'out_mon_reg_y_exp.pickle')
    et.save(out_mon_irreg_z,'out_mon_reg_z_exp.pickle')
    np.savetxt(d+'input_frequencies_exp.txt',frequencies)
    et.save(inputpop,'inputpop_exp.pickle')
    chip.save_parameters(d+'biases_chip')
   
    binning = 100000.0
    time_plot_freq = np.linspace(0,max((dy-min(dy)+800)),max((dy-min(dy)+800))/binning)

    #regular
    neuron_activity_rex_z = compute_neuron_activity(out_mon_reg_z.sl, time_plot_freq)
    neuron_activity_rex_x = compute_neuron_activity(out_mon_reg_x.sl, time_plot_freq)
    neuron_activity_rex_y = compute_neuron_activity(out_mon_reg_y.sl, time_plot_freq)

    #irregular 
    neuron_activity_irrex_z = compute_neuron_activity(out_mon_irreg_z.sl, time_plot_freq)
    neuron_activity_irrex_x = compute_neuron_activity(out_mon_irreg_x.sl, time_plot_freq)
    neuron_activity_irrex_y = compute_neuron_activity(out_mon_irreg_y.sl, time_plot_freq)

    figure()
    plot(time_plot_freq[1:-1:]*1e-3,neuron_activity_rex_x[1:-1:],'ro-', label='regular x')
    hold(True)
    plot(time_plot_freq[1:-1:]*1e-3,neuron_activity_rex_y[1:-1:],'bo-', label='regular y')
    grid(True)
    plot(time_plot_freq[1:-1:]*1e-3,neuron_activity_rex_z[1:-1:],'go-',label='regular z')    
    hold(True)
    xlabel(r'time [ms]', fontsize=22)
    ylabel(r'$\nu_{out}$ [Hz]', fontsize=22)
    legend(loc='upper left')
    filename = str(figure_directory)+str(figure_counter)+str('_output_sensor_data_regular.png')
    savefig(filename)     
    figure_counter += 1

    figure()
    plot(time_plot_freq[1:-1:]*1e-3,neuron_activity_irrex_x[1:-1:],'ro-', label='irregular x')
    hold(True)
    plot(time_plot_freq[1:-1:]*1e-3,neuron_activity_irrex_y[1:-1:],'bo-', label='irregular y')
    grid(True)
    plot(time_plot_freq[1:-1:]*1e-3,neuron_activity_irrex_z[1:-1:],'go-', label='irregular z')    
    hold(True)
    xlabel(r'time [ms]', fontsize=22)
    ylabel(r'$\nu_{out}$ [Hz]', fontsize=22)
    legend(loc='upper left')
    filename = str(figure_directory)+str(figure_counter)+str('_output_sensor_data_irregular.png')
    savefig(filename)     
    figure_counter += 1
   
    return


def compute_neuron_activity(outmon, frequencies):
    
    #make everything raw data
    spikeout = outmon
    rawOut = spikeout.raw_data()
    activeNeu = np.unique(rawOut[:,1])
    activeNeu = activeNeu.astype('int')
    
    neuronActivity = []
    tstart =  min(rawOut[:,0])
    tstop = max(rawOut[:,0])
    nslice = len(frequencies)
    binningRec = (tstop - tstart)/ nslice
    timeAxis = []
    #calculate mean activity for every slice of stimulation at a different frequency
    for i in range(nslice):
        timeAxis.append(binningRec*(i+1))
        tmppostFreqs = meanNeuFiring(rawOut,activeNeu,[min(rawOut[:,0])+binningRec*i,min(rawOut[:,0])+binningRec*(i+1)])
        neuronActivity.append(tmppostFreqs)

    return neuronActivity

def measurePhy_1(d, setup,chip,mypop,inputpop,neu,syn, out_mon):

    figure_counter = int(time()) 
    figure_directory = d + 'figures/'
    if os.path.isdir(figure_directory):
        print 'figure directory already exists..'
    else:
        os.mkdir(figure_directory)

    chip.load_parameters('biases/new_vestibular_adapt.txt')
    #chip.load_parameters('biases/cv_irregular_neuron.txt')

    chip.set_parameter('ifdc',1.8)#71
    chip.set_parameter('iftau',0.17)
    chip.set_parameter('ifthr',0.375) #66
    chip.set_parameter('ifrfr',0.3)

    chip.set_parameter('dpitau',1.628)
    chip.set_parameter('dpithr',1.478) #irregular if 1.49 --> regular 1.46

    chip.set_parameter('whi',0.5)
    chip.set_parameter('wthr',0.28) #0.28
    
    chip.set_parameter('ahtau',0)
    chip.set_parameter('ahw',1.2)   #no adapt 1.8
    chip.set_parameter('athr',0.08)  #no adapt 0 
    chip.set_parameter('ifcasc',0.3) # no adapt 0

    chip.set_parameter('ifdc',1.8)    
    chip.set_parameter('pwlk',1.5) 

    #setup the stimulation protocol
    stepSize = 500
    durations =  np.arange(0,20000,1000)#np.arange(0,20000,1000)  np.arange(0,7500,500)#    
    minFreq = 0
    maxFreq = 85
    nstepf = 20
    frequencies = np.linspace(minFreq,maxFreq,nstepf)
    #[1,10,20,42,65,80,90,108,110,120,130,140,150,160,170,180,200,220,250,280];#  [50,55,60,65,70]#   

    spikeTrainLearnSin = []
    #set up the train stimulus learn syn
    for i in range(len(frequencies)):
        spikeTrain = inputpop[neu].synapses['learning'][syn].spiketrains_regular(frequencies[i],t_start=durations[i],duration=stepSize)
        spikeTrainLearnSin.append(spikeTrain)

    #merge spikelists  
    totspkLearn = pyNCS.pyST.merge_sequencers(*spikeTrainLearnSin)
    
	#stimulate the chip and record spiketrain in the monitors
    x = setup.stimulate(totspkLearn, send_reset_event=False, tDuration=max(durations))
	
	#make everything raw data
    spikeout = out_mon.sl
    rawOut = spikeout.raw_data()
    activeNeu = np.unique(rawOut[:,1])
    activeNeu = activeNeu.astype('int')

    np.savetxt(d+'out_mon_raw_data.txt',rawOut)
    
    neuronActivity = []
    tstart =  min(rawOut[:,0])
    tstop = max(rawOut[:,0])
    nslice = len(frequencies)
    binningRec = (tstop - tstart)/ nslice
    timeAxis = []
	#calculate mean activity for every slice of stimulation at a different frequency
    for i in range(nslice):
        timeAxis.append(binningRec*(i+1))
        tmppostFreqs = meanNeuFiring(rawOut,activeNeu,[min(rawOut[:,0])+binningRec*i,min(rawOut[:,0])+binningRec*(i+1)])
        indexOurNeu = np.nonzero(activeNeu==neu)
        neuronActivity.append(tmppostFreqs[indexOurNeu])

	#plot the transfer function
    figure(1)
    plot(frequencies[0:-1:],neuronActivity[0:-1:],'o--')
    hold(True)
    #bisect = np.arange(0,max(outMeanFiringRates[0:-1:]),0.5);
    #plot(bisect,bisect)
    grid(True)
    xlabel('Input Freq [Hz]', fontsize=18)
    ylabel('Output Freq [Hz]', fontsize=18)
    filename = str(figure_directory)+str(figure_counter)+str('_fVsf.png')
    savefig(filename)     
    figure_counter += 1

    #now lets make the histogram regular
    neuToStim = neu+1
    chip.set_parameter('dpithr',1.38) #it will fire around 100 Hz
    spikeTrain = inputpop[neuToStim].synapses['learning'][syn].spiketrains_regular(1,t_start=0,duration=20000)
    x = setup.stimulate(spikeTrain, send_reset_event=False, tDuration=20000)
    
    figure(2)
    spikeoutReg = out_mon.sl
    rawOut = spikeoutReg.raw_data()
    isi_reg = np.diff(rawOut[:,0])*1e-3
    np.savetxt(d+'out_mon_raw_data_isi_reg.txt',isi_reg)
    hist(isi_reg,28,normed=True,range=(0, 0.03))
    xlabel('ISI [s]',fontsize=18)
    ylabel('counts', fontsize=18)
    filename = str(figure_directory)+str(figure_counter)+str('_isi_reg.png')
    savefig(filename)     
    figure_counter += 1
 
    #now lets make the histogram irregular
    neuToStim = neu
    chip.set_parameter('dpithr',1.478) 
    popTotrain = inputpop[neu].synapses['learning'][syn]
    spikeTrain = popTotrain.spiketrains_regular_gaussian(300,scale=1000,t_start=0,duration=10000)
    #spikeTrain = popTotrain.spiketrains_poisson(100,t_start=0, duration= 20000)
    x = setup.stimulate(spikeTrain, send_reset_event=False, tDuration=20000)
    
    figure(3)
    spikeoutIrr = out_mon.sl
    rawOut = spikeoutIrr.raw_data()
    isi_irreg = np.diff(rawOut[:,0])*1e-3
    indextokeep = np.nonzero(isi_irreg < 0.03)   #this is related to the aex
    okisi_irreg = isi_irreg[indextokeep]
    np.savetxt(d+'out_mon_raw_data_isi_irreg.txt',okisi_irreg)
    hist(okisi_irreg,28,normed=True,range=(0, 0.03))
    xlabel('ISI [s]',fontsize=18)
    ylabel('counts', fontsize=18)
    filename = str(figure_directory)+str(figure_counter)+str('_isi_irreg.png')
    savefig(filename)     
    figure_counter += 1
 
    show()
    #plot hist syn efficacy


    return


def measurePhy(setup,chip,mypop,inputpop,neu,syn):


    chip.load_parameters('biases/new_vestibular_adapt.txt')

    chip.set_parameter('ifdc',1.8)#71
    chip.set_parameter('iftau',0.17)
    chip.set_parameter('ifthr',0.375) #66
    chip.set_parameter('ifrfr',0.3)

    chip.set_parameter('dpitau',1.628)
    chip.set_parameter('dpithr',1.478) #irregular if 1.49 --> regular 1.46

    chip.set_parameter('whi',0.1)
    chip.set_parameter('wthr',0.1) #0.28
    
    chip.set_parameter('ahtau',0)
    chip.set_parameter('ahw',1.2)   #no adapt 1.8
    chip.set_parameter('athr',0.08)  #no adapt 0 
    chip.set_parameter('ifcasc',0.3) # no adapt 0


    chip.set_parameter('ifdc',1.8)    
    chip.set_parameter('pwlk',1.5) 

    #setup the stimulation protocol
    stepSize = 500
    durations =  np.arange(0,20000,1000)#np.arange(0,20000,1000)  np.arange(0,7500,500)#    
    minFreq = 350
    maxFreq = 1500
    nstepf = 20
    frequencies = np.linspace(minFreq,maxFreq,nstepf)
    #[1,10,20,42,65,80,90,108,110,120,130,140,150,160,170,180,200,220,250,280];#  [50,55,60,65,70]#   

    spikeTrainLearnSin = []
    #set up the train stimulus learn syn
    for i in range(len(frequencies)):
        spikeTrain = inputpop[neu].synapses['learning'][syn].spiketrains_regular(frequencies[i],t_start=durations[i],duration=stepSize)
        spikeTrainLearnSin.append(spikeTrain)

    #merge spikelists  
    totspkLearn = pyNCS.pyST.merge_sequencers(*spikeTrainLearnSin)
    
    #monitors 
    outMon = pyNCS.monitors.SpikeMonitor(mypop.soma[neu])
    inMon = pyNCS.monitors.SpikeMonitor(inputpop.soma[syn])  
    setup.monitors.import_monitors([outMon,inMon])

	#stimulate the chip and record spiketrain in the monitors
    x = setup.stimulate(totspkLearn, send_reset_event=False, tDuration=max(durations))
	
	#make everything raw data
    spikeout = outMon.sl
    rawOut = spikeout.raw_data()
    activeNeu = np.unique(rawOut[:,1])
    activeNeu = activeNeu.astype('int')

    neuronActivity = []
    tstart =  min(rawOut[:,0])
    tstop = max(rawOut[:,0])
    nslice = len(frequencies)
    binningRec = (tstop - tstart)/ nslice
    timeAxis = []
	#calculate mean activity for every slice of stimulation at a different frequency
    for i in range(nslice):
        timeAxis.append(binningRec*(i+1))
        tmppostFreqs = meanNeuFiring(rawOut,activeNeu,[min(rawOut[:,0])+binningRec*i,min(rawOut[:,0])+binningRec*(i+1)])
        indexOurNeu = np.nonzero(activeNeu==neu)
        neuronActivity.append(tmppostFreqs[indexOurNeu])

	#plot the transfer function
    ion()
    figure(1)
    plot(frequencies[0:-1:],neuronActivity[0:-1:],'o--')
    hold(True)
    #bisect = np.arange(0,max(outMeanFiringRates[0:-1:]),0.5);
    #plot(bisect,bisect)
    grid(True)
    xlabel('Input Freq [Hz]', fontsize=18)
    ylabel('Output Freq [Hz]', fontsize=18)


    #now lets make the histogram regular
    neuToStim = neu+1
    chip.set_parameter('dpithr',1.38) #it will fire around 100 Hz
    spikeTrain = inputpop[neuToStim].synapses['learning'][syn].spiketrains_regular(1,t_start=0,duration=10000)
    x = setup.stimulate(spikeTrain, send_reset_event=False, tDuration=10000)
    
    figure(2)
    spikeoutReg = outMon.sl
    rawOut = spikeoutReg.raw_data()
    isi_reg = np.diff(rawOut[:,0])*1e-3
    np.savetxt('results/isi_reg.txt',isi_reg)
    hist(isi_reg,28,range=(0, 0.03),normed=True)
    xlabel('ISI [s]',fontsize=18)
    ylabel('counts', fontsize=18)

    #now lets make the histogram irregular
    neuToStim = neu
    chip.set_parameter('dpithr',1.478) 
    popTotrain = inputpop[neu].synapses['learning'][syn]
    spikeTrain = popTotrain.spiketrains_regular_gaussian(300,t_start=0,duration=10000)
    x = setup.stimulate(spikeTrain, send_reset_event=False, tDuration=10000)
    
    figure(3)
    spikeoutIrr = outMon.sl
    rawOut = spikeoutIrr.raw_data()
    isi_irreg = np.diff(rawOut[:,0])*1e-3
    indextokeep = np.nonzero(isi_irreg < 0.03)   #this is related to the aex
    okisi_irreg = isi_irreg[indextokeep]
    np.savetxt('results/isi_irreg.txt',okisi_irreg)
    hist(okisi_irreg,28,range=(0, 0.03),normed=True)
    xlabel('ISI [s]',fontsize=18)
    ylabel('counts', fontsize=18)

    show()
    #plot hist syn efficacy


    return


def inputVestibuar_final():
    #lets make a network of 6 neurons
    #acceleration excitation - deceleration inhibition 
    #regular afferent
    regular_x_pos = np.array([0])
    regular_y_pos = np.array([1])
    regular_z_pos = np.array([2])

    #irregular afferent
    irregular_x_pos = np.array([3])
    irregular_y_pos = np.array([4])
    irregular_z_pos = np.array([5])

    #load biases
    chip.load_parameters('biases/new_vestibular_adapt.txt')

    return

def inputVestibular(setup,chip,mypop,inputpop,neu,syn):


    #chip.load_parameters('biases/withQN.txt')
    #chip.set_parameter('ifthr',(0.3/1.8)*3.3)
    #chip.set_parameter('dpitau',(1.608/1.8)*3.3)
    #chip.set_parameter('ifrfr',(0.3/1.8)*3.3)
    #chip.set_parameter('ahw',(0.2/1.8)*3.3)
    #chip.set_parameter('ahtau',(0.35/1.8)*3.3)

    #chip.load_parameters('biases/biasSpikebetternofiring_adapt.txt')
    #chip.set_parameter('wdriftdn',(0/1.8)*3.3)
    #chip.set_parameter('wdriftup',(1.7/1.8)*3.3)
    #chip.set_parameter('deltadn', (0/1.8)*3.3)
    #chip.set_parameter('deltaup', (0/1.8)*3.3)
    #setSynState(neu,syn,1)
    #adapt
    #chip.set_parameter('ahw',(0.2/1.8)*3.3)
    #chip.set_parameter('ahtau',(0.27/1.8)*3.3)
    #chip.set_parameter('athr',(0.8/1.8)*3.3)
    #chip.set_parameter('ifcasc',(0.144/1.8)*3.3)

    #200 Hz
    chip.load_parameters('biases/new_vestibular_adapt.txt')

    chip.set_parameter('ifdc',1.8)#71
    chip.set_parameter('iftau',0.17)
    chip.set_parameter('ifthr',0.375) #66
    chip.set_parameter('ifrfr',0.3)

    chip.set_parameter('dpitau',1.6)
    chip.set_parameter('dpithr',1.4)

    chip.set_parameter('whi',0.1)
    chip.set_parameter('wthr',0.28)
    
    chip.set_parameter('ahtau',0)
    chip.set_parameter('ahw',1.2)   #no adapt 1.8
    chip.set_parameter('athr',0.08)  #no adapt 0 
    chip.set_parameter('ifcasc',0.3) # no adapt 0

    chip.set_parameter('ifdc',1.46)

    #chip.load_parameters('biases/irregular_neuron_bursting.txt')

    ins = open( "sensor_data/GyroReadings.txt", "r" )
    array = []
    count = 0
    for line in ins:
        array.append( line )
        count = count+1;

    #extract data into array
    dx = np.zeros(count)
    dy = np.zeros(count)
    dz = np.zeros(count)
    dms = np.zeros(count)
    dns = np.zeros(count)
    for index, item in enumerate(array):
        dx[index],dy[index],dz[index],dms[index],dns[index] = item.rsplit(',')


    #normalize 
    ndx = dx/abs(dx).max()
    ndx = ndx+abs(min(ndx))

    ndy = dy/abs(dy).max()
    ndy = ndy+abs(min(ndy))

    ndz = dz/abs(dz).max()
    ndz = ndz+abs(min(ndz))

    #we create a train
    binning = 10   #ms stimulation per variation
    nslice = 240   #number of slice of the original signal

    binningOrig = max(dms)/nslice
    durationOrig = max(dms)

    totalDuration = binning * nslice #total duration of stimulation
    fakeBinning =  totalDuration / nslice   

    neu = 0 
    syn = 0
    multiplyFactor = 100 #frequency scaling

    outMon = pyNCS.monitors.SpikeMonitor(mypop.soma)
    inMon = pyNCS.monitors.SpikeMonitor(inputpop.soma)  
    setup.monitors.import_monitors([outMon, inMon])


    timebins = np.linspace(0, durationOrig, size(ndz))
    rate = ndz*multiplyFactor
    islearning = inputpop[neu].synapses['learning'][syn]
    spiketrain = islearning.spiketrains_inh_poisson(r_[[rate]],timebins)#[0]
    
    x = setup.stimulate(spiketrain, send_reset_event=False, tDuration=max(timebins))

    spikeout = outMon.sl
    rawOut = spikeout.raw_data()
    activeNeu = np.unique(rawOut[:,1])
    activeNeu = activeNeu.astype('int')

    neuronActivity = []
    tstart =  min(rawOut[:,0])
    tstop = max(rawOut[:,0])
    nslice = 30
    binningRec = (tstop - tstart)/ nslice
    timeAxis = []
    for i in range(nslice):
        timeAxis.append(binningRec*(i+1))
        tmppostFreqs = meanNeuFiring(rawOut,activeNeu,[min(rawOut[:,0])+binningRec*i,min(rawOut[:,0])+binningRec*(i+1)])
        indexOurNeu = np.nonzero(activeNeu==neu)
        neuronActivity.append(tmppostFreqs[indexOurNeu])
    
    #plot
    ion()

    figure(2)
    plot(dz)
    xlabel(r'$time$ [ms]',fontsize=18)
    ylabel(r'x',fontsize=18)

    figure(3)
    plot(timeAxis,neuronActivity,'o--')
    xlabel(r'$time$ [ms]',fontsize=18)
    ylabel(r'$\nu$ [Hz]',fontsize=18)

    #totspk[0].raster_plot()
    outMon.raster_plot()


    return 

