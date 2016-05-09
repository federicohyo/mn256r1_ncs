delta=50
n_spikes=5
neu_sync = 5
freq_sync = 400
duration_sync = 500
delay_sync = 700


self = sl

def stimulate(stimulus, **kwargs):
    stim_evs = nsetup._pre_process(stimulus)
    #evs = nsetup.communicator.run_rec(stim_evs, **kwargs)
    stim_fn, mon_fn = gen_rec_fns()
    save_rec_file(stim_fn, stim_evs,header = nsetup.communicator.REC_HEADER_SEQ) 
    
    mon_evs = nsetup.run(stimulus)
    
    nsetup._post_process(evs, nsetup.monitors.channels)
    
def save_rec_file(ev_array,stim_fn, *args, **kwargs):    
    import numpy
    if int(numpy.__version__.split('.')[0])==1 and int(numpy.__version__.split('.')[1])<=6:
         kwargs.pop('header')
    nsetup.communicator._rec_fns.append(filename)
    numpy.savetxt(filename, stim_fn, delimiter = ' ', newline = '\n', fmt = ['%u', '%u'], *args, **kwargs)
    nsetup.communicator._run_id += 1
    
    
def gen_rec_fns():
    '''
    Generate filenames for recording
    '''
    import time
    N = nsetup.communicator._run_id
    current_time_str = time.strftime("__" + "%d-%m-%Y-%H:%M:%S", time.localtime())
    filename = nsetup.communicator.REC_PATH + '_{2}_{0}__run{1}.dat'
    stim_fn = filename.format(current_time_str, N, nsetup.communicator.REC_FN_SEQ)
    mon_fn = filename.format(current_time_str, N, nsetup.communicator.REC_FN_MON )
    return stim_fn, mon_fn
    

syn_virtual = inputpop.synapses['virtual_exc']
stim_timing = syn_virtual.spiketrains_poisson(40,duration=1000)
out = nsetup.stimulate(stim_timing)

somach = self.neurons.soma.channel
sns = self.neurons.synapses[self.syntype]
snsoma = self.neurons.soma
rates = 1./delta
duration = (delta * self.y_dim) + delay_sync 
wij = np.zeros((self.x_dim, self.y_dim))
#enable broadcast 
a = range(256)
broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][0::self.y_dim]
stim = broadcast_syn.spiketrains_regular(rates, offset=range(delay_sync, delay_sync+(self.y_dim*delta)+delta, delta), duration=duration)
#program broadcast syn
matrix_b = np.ones([256,256])
self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
self.setup.mapper._program_onchip_broadcast_learning(matrix_b)
#load biases
self.setup.chips['mn256r1'].load_parameters('biases/biases_wijget.biases')
#create sync spike train
index_neu = self.pop_broadcast.synapses['virtual_exc'].addr['neu'] == neu_sync
syn_sync = self.pop_broadcast.synapses['virtual_exc'][index_neu]
sync_spikes = syn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
#merge sync train and broadcast train
stimulus = pyNCS.pyST.merge_sequencers(sync_spikes, stim)
#stimulate
index_neu = self.pop_broadcast.synapses['virtual_exc'].addr['neu'] == neu_sync+1
nsyn_sync = self.pop_broadcast.synapses['virtual_exc'][index_neu]
nsync_spikes = nsyn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
self.setup.stimulate({},send_reset_event=True,duration=duration_sync+500)#clear eventual spikes burst
time.sleep(1.2)
all_neu = pyNCS.Population("","")
all_neu.populate_all(self.setup,'mn256r1','excitatory')
monitor = pyNCS.monitors.SpikeMonitor(all_neu.soma)
nsetup.monitors.import_monitors([monitor])
out = self.setup.stimulate(stimulus,send_reset_event=False,duration=duration+1500)
time.sleep(1.2)
out = out[somach]
#clean data
raw_data = out.raw_data()
sync_index = raw_data[:,1] == neu_sync
start_time = np.min(raw_data[sync_index,0])
br_start = start_time + delay_sync - 2 
br_stop = br_start + (delta*self.y_dim)
out.t_start = br_start 
out.t_stop = br_stop
this_f = out.firing_rate(delta) 
self.state = 1*(this_f > 0)
#off broadcast
matrix_b[:] = 0
self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
self.setup.mapper._program_onchip_broadcast_learning(matrix_b)
