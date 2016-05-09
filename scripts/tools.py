
def probeNeuSyn(neuron,synapse,chip,setup,inputpop):
    #set up the scope for monitoring
    print('Preparing to probe membrane potential of neuron ')
    chip.set_parameter('probe_reset_dac',1.8)
    chip.set_parameter('probe_reset_dac',0)
    chip.set_parameter('nlatch_voltage2',0)

    #set up the train stimulus exc1 syn
    spikeTrain = inputpop[neuron].synapses['learning'][synapse].spiketrains_regular(1.0,t_start=0,duration=1000)
    setup.stimulate(spikeTrain)

    chip.set_parameter('nlatch_voltage2',1.8)
    print('Vmem of target neuron available on pad ifVmem_mon')

    print('now we probe the synapse')
    chip.set_parameter('nlatch_voltage1',0)
    spikeTrain = inputpop[neuron].synapses['learning'][synapse].spiketrains_regular(1.0,t_start=0,duration=1000)
    setup.stimulate(spikeTrain)
    chip.set_parameter('nlatch_voltage1',1.8)
    print('Vw of target synapse available on pad Vw_mon')



