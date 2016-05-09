

per_spike_time = monitor_per.sl.raw_data()[:,0]
raw_rcn = monitor_rcn.sl.raw_data()


f_test_syn = 50
syn_learning_perc = pop_perceptrons.synapses['learning']
stim_test_syn = syn_learning_perc.spiketrains_regular(f_test_syn, jitter=True, duration=15000)

out = nsetup.stimulate(stim_test_syn,send_reset_event=False)
