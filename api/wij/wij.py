import numpy as np
import pickle
from pylab import *
import pyNCS
from pyNCS.pyST import SpikeList
import time
from scipy.interpolate import griddata
import pyAex

class SynapsesLearning():
    """
    Represents the matrix of synaptic weights of a chip.
    """
    def __init__(self, population, syntype='learning', shape=None):
        """
        """
        addrgroup = population.synapses[syntype]
        self.neurons = population
        self.syntype = syntype
        self.setup = population.setup
        if shape is None:
            # guess the shape (x = neuron, y = synapse)
            field0 = addrgroup.dtype.descr[0][0]
            self.x_dim = len(np.unique(addrgroup.addr[field0]))
        self.state = np.zeros(len(addrgroup)).reshape(self.x_dim, -1)
        self.y_dim = self.state.shape[1]
        self.pop_broadcast = pyNCS.Population("","")
        self.pop_broadcast.populate_all(self.setup, 'mn256r1', 'excitatory')        
        self.out_mon = pyNCS.monitors.SpikeMonitor(self.pop_broadcast.soma)
        self.setup.monitors.import_monitors([self.out_mon])


    def do_pltp(self, min_freq = 0, max_freq=500, nsteps =5, duration = 1000):
        '''
        set learning matrix to all zeros and stimulate
        return input/output matrices
        '''
        #enable broadcast plastic
        matrix_b = np.ones([256,256])
        self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
        self.setup.mapper._program_onchip_broadcast_learning(matrix_b)

        a = range(256)
        broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][0::self.y_dim]
        freqs = np.linspace(min_freq, max_freq, nsteps)
        wijs_final = []
        wijs_init = []
        for f in freqs:
            set_rand = np.zeros([256,256]) 
            wijs_init.append(set_rand)
            self.set(set_rand)
            self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
            self.setup.mapper._program_onchip_broadcast_learning(matrix_b)
            self.setup.chips['mn256r1'].load_parameters('biases/biases_wijlearning.biases')
            time.sleep(1)
            this_stim = broadcast_syn.spiketrains_poisson(f,duration=duration)
            out = self.setup.stimulate(this_stim, send_reset_event=False, duration = duration)
            self.get_br()
            wijs_final.append(self.state)

        return freqs, wijs_init, wijs_final

    def do_pltd(self, min_freq = 0, max_freq=500, nsteps =5, duration = 1000):
        '''
        set learning matrix to all ones and stimulate
        return input/output matrices
        '''
        #enable broadcast plastic
        matrix_b = np.ones([256,256])
        self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
        self.setup.mapper._program_onchip_broadcast_learning(matrix_b)

        a = range(256)
        broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][0::self.y_dim]
        freqs = np.linspace(min_freq, max_freq, nsteps)
        wijs_final = []
        wijs_init = []
        for f in freqs:
            set_rand = np.ones([256,256]) 
            wijs_init.append(set_rand)
            self.set(set_rand)
            self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
            self.setup.mapper._program_onchip_broadcast_learning(matrix_b)
            self.setup.chips['mn256r1'].load_parameters('biases/biases_wijlearning.biases')
            time.sleep(1)
            this_stim = broadcast_syn.spiketrains_poisson(f,duration=duration)
            out = self.setup.stimulate(this_stim, send_reset_event=False, duration = duration)
            self.get_br()
            wijs_final.append(self.state)

        return freqs, wijs_init, wijs_final

    def do_pltp_pltd_vs_fpre_fpost(self, min_freq = 0, max_freq= 350, nsteps = 10, duration=600, min_inj = 6e-08, max_inj=190e-8 , nsteps_injs = 3, init_syn = 2):
        '''
        make measurements for plains
        init_syn = 1 -> all learning synapses initialized with state = 1
        init_syn = 0 -> all learning synapses initialized with state = 0
        init_syn = 2 -> random init learning synapses
        '''
        #enable broadcast plastic
        matrix_b = np.ones([256,256])
        self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
        self.setup.mapper._program_onchip_broadcast_learning(matrix_b)

        somach = self.neurons.soma.channel
        #create stim broadcast
        a = range(256)
        broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][0::self.y_dim]
        freqs = np.linspace(min_freq, max_freq, nsteps)
        injs = np.linspace(min_inj, max_inj, nsteps_injs)
        wijs_final = []
        wijs_init = []
        post_freqs = []
        pre_freqs = []
        for this_inj in range(len(injs)):
            for f in freqs:
                if(init_syn == 2):
                    set_rand = 1*( np.random.random([256,256]) < 0.5)
                elif(init_syn == 1):
                    set_rand = np.ones([256,256])
                elif(init_syn == 0):
                    set_rand = np.zeros([256,256])
                wijs_init.append(set_rand)
                self.set(set_rand)
                self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
                self.setup.mapper._program_onchip_broadcast_learning(matrix_b)
                self.setup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_ret.biases')
                self.setup.chips['mn256r1'].configurator.set_parameter("IF_DC_P", injs[this_inj])
                time.sleep(0.5)
                this_stim = broadcast_syn.spiketrains_poisson(f,duration=duration)
                out = self.setup.stimulate(this_stim, send_reset_event=False, duration = duration)
                time.sleep(0.3)
                out = out[somach]
                start = np.max(out.raw_data()[:,0])-duration
                stop = start + duration
                out.t_start = start
                out.t_stop = stop
                neu_f = out.mean_rates()
                this_post_f_matrix = np.repeat(neu_f, self.y_dim).reshape(self.y_dim,self.y_dim)
                out,stim,state = self.get_br(debug=True)
                wijs_final.append(self.state)
                this_pre_f_matrix = np.repeat(f,self.y_dim*self.y_dim).reshape([self.y_dim,self.y_dim])
                pre_freqs.append(this_pre_f_matrix)
                post_freqs.append(this_post_f_matrix)

        return pre_freqs, post_freqs, wijs_init, wijs_final



    def do_pltp_pltd_vs_fpre(self, min_freq=0, max_freq=350, nsteps=10, duration=1000):
        '''
        '''
        #enable broadcast plastic
        matrix_b = np.ones([256,256])
        self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
        self.setup.mapper._program_onchip_broadcast_learning(matrix_b)

        #create stim broadcast
        a = range(256)
        broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][0::self.y_dim]
        freqs = np.linspace(min_freq, max_freq, nsteps)
        wijs_final = []
        wijs_init = []
        for f in freqs:
            set_rand = 1*( np.random.random([256,256]) < 0.5)
            wijs_init.append(set_rand)
            self.set(set_rand)
            self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
            self.setup.mapper._program_onchip_broadcast_learning(matrix_b)
            self.setup.chips['mn256r1'].load_parameters('biases/biases_wijlearning.biases')
            this_stim = broadcast_syn.spiketrains_poisson(f,duration=duration)
            out = self.setup.stimulate(this_stim, send_reset_event=False, duration = duration)
            self.get_br()
            wijs_final.append(self.state)

        return freqs, wijs_init, wijs_final


    def plot_plane_ltp_ltd(self, pre_f, post_f, wij_inputs, wij_outputs, plane = 2):
        '''
        do the plain plot
        plane = 1 -> plot ltp
        plane = 0 -> plot ltd
        plane = 2 -> plot both
        '''
        nsyn_tot = self.y_dim*self.y_dim
        ntrial = len(pre_f)
        fpre_ltp = []
        fpost_ltp = []
        prob_ltp = []
        prob_ltd = []
        fpre_ltd = []
        fpost_ltd = []
        for this_trial in range(ntrial):
            fpre = np.mean(pre_f[this_trial])
            fposts = post_f[this_trial][:,0]
            #cicle over each post neurons (post_freq)
            for this_post in range(self.y_dim):
                learn_syn = np.where(wij_outputs[this_trial][this_post,:]-wij_inputs[this_trial][this_post,:]!=0)[0]
                syn_up = len(np.where(wij_inputs[this_trial][this_post,:] == 1)[0])
                syn_dn = len(np.where(wij_inputs[this_trial][this_post,:] == 0)[0])
                ltp_syn = np.where((wij_outputs[this_trial][this_post,learn_syn] - wij_inputs[this_trial][this_post,learn_syn]) > 0)[0]
                ltd_syn = np.where((wij_outputs[this_trial][this_post,learn_syn] - wij_inputs[this_trial][this_post,learn_syn]) < 0)[0]
                #save freqs and probs
                fpre_ltd.append(fpre)
                fpost_ltd.append(fposts[this_post])
                fpre_ltp.append(fpre)
                fpost_ltp.append(fposts[this_post])
                nsyn = len(ltd_syn)
                if(syn_up > 0):
                    prob_ltd.append(float(nsyn)/syn_up)
                else:
                    prob_ltd.append(0)
                nsyn = len(ltp_syn)
                if(syn_dn > 0):
                    prob_ltp.append(float(nsyn)/syn_dn)
                else:
                    prob_ltp.append(0)

        if(plane == 0 or plane == 2):              
            figure() 
            x=np.asarray(fpre_ltd)
            y=np.asarray(fpost_ltd)
            z= np.asarray(prob_ltd)
            bigg  = np.max(x)
            bigg = [bigg, np.max(y)]
            ming = np.min(x)
            ming = [ming, np.min(y)]
            steps = np.linspace(0.1,220,10)
            X , Y = np.meshgrid(steps,steps)
            ZI = griddata((x,y),z,(X, Y), method='cubic');
            CS3 = contour(X, Y, ZI, 10,extend='both')
            clabel(CS3, fmt = '%2.1f', colors = 'b', fontsize=12)
            xlabel(r"$\nu_{pre} [Hz]$", fontsize=18)
            ylabel(r"$\nu_{post} [Hz]$", fontsize=18)
            title("P(ltd)")
            
        if(plane == 1 or plane == 2):  
            figure()
            x=np.asarray(fpre_ltp)
            y=np.asarray(fpost_ltp)
            z= np.asarray(prob_ltp)
            bigg  = np.max(x)
            bigg = [bigg, np.max(y)]
            ming = np.min(x)
            ming = [ming, np.min(y)]
            steps = np.linspace(0.1,220,10)
            X , Y = np.meshgrid(steps,steps)
            ZI = griddata((x,y),z,(X, Y), method='cubic');
            CS3 = contour(X, Y, ZI, 10,extend='both')
            clabel(CS3, fmt = '%2.1f', colors = 'b', fontsize=12)
            xlabel(r"$\nu_{pre} (Hz)$", fontsize=18)
            ylabel(r"$\nu_{post} (Hz)$", fontsize=18)
            title("P(ltp)")
             

    def calc_ltp_ltd_probs(self, freqs, inputs, outputs, do_plot = False):
        '''
        plot input output probabilities
        '''
        nsyn_tot = self.y_dim*self.x_dim
        nsteps = len(freqs)
        pltp = [] 
        pltd = []
        for this_step in range(nsteps):
            if(np.shape(outputs[this_step]) == np.shape(inputs[this_step])):
                diff_m = outputs[this_step] - inputs[this_step]
                ltp_num = len(np.where(diff_m == 1)[0])
                ltd_num = len(np.where(diff_m == -1)[0])
                changed_syn = len(np.where(diff_m != 0)[0])
                pltp.append(ltp_num/(256*256.0))
                pltd.append(ltd_num/(256*256.0))
            else:
                print "we skip a step"
                pltp.append(-1)
                pltd.append(-1)

        if(do_plot == True):
            figure()
            plot(freqs, pltd, 'o-', label='P(LTD)')
            plot(freqs, pltp, 'o-', label='P(LTP)')
            xlabel(r'$\nu_{pre}$ [Hz]')
            ylabel(r'P(ltp),P(ltd)')
            legend(loc='best')

        return freqs, pltp, pltd

    def _stimulate(self, stimulus, duration, channel = 0, host='localhost', port=50001, fps=25):
        '''
        no matter how it will work...
        '''
        monchaddress = pyNCS.pyST.getDefaultMonChannelAddress()
        pyNCS.pyST.STas.addrBuildHashTable(monchaddress[channel])
        eventsQueue = pyAex.aexclient.AEXMonClient(MonChannelAddress=monchaddress, 
                                                        channels=[channel],
                                                        host=host,
                                                        port=port,
                                                        autostart=True,
                                                        fps=fps)
        time_len = 0
        #listen and stimulate for duration
        while time_len < duration:
            eventsPacket = eventsQueue.buffer.get(block=False)
            add_events = eventsPacket.get_ad()
            time_events = eventsPacket.get_tm()
            index_to_get = (add_events < 256)


    def get_br_timos(self, delta=50, n_spikes=5, neu_sync = 5, freq_sync = 400, duration_sync = 500, delay_sync = 700, debug=False):
        endsync_sync = self.pop_broadcast.synapses['virtual_exc'][index_neu]
        endsync_spikes = endsync_sync.spiketrains_regular(freq_sync, duration=duration_sync, t_start=time_stop)
        stimulus = pyNCS.pyST.merge_sequencers(stimulus, endsync_spikes)
        #stimulate
        time.sleep(1.5)
        nindex_neu = self.pop_broadcast.synapses['virtual_exc'].addr['neu'] == (neu_sync+2)
        nsyn_sync = self.pop_broadcast.synapses['virtual_exc'][nindex_neu]
        nync_spikes = nsyn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
        out = self.setup.stimulate(nync_spikes,send_reset_event=True)
        out = self.setup.stimulate(nync_spikes,send_reset_event=False) #should be nice to know why...
        out = self.setup.stimulate(stimulus, send_reset_event=False, duration=duration+duration_sync)
        time.sleep(1.5)
        out = out[somach]
        #clean data
        self.raw_data = out.raw_data()
        neur_index = []
        for i in range(self.y_dim):
            neur_index.append(self.raw_data[:, 1] == i)
        sync_index = self.raw_data[:, 1] == neu_sync
        others_index = self.raw_data[:, 1] != neu_sync
        gap = 0
        for i in self.raw_data[sync_index, :].T[0]:
            sync_start = i
            earlier_otherneuron_spikes_index = (self.raw_data.T[0] < i) * others_index
            if self.raw_data[earlier_otherneuron_spikes_index, :].T[0].any():
                last_earlier_otherneuron_spike_time = np.max(self.raw_data[earlier_otherneuron_spikes_index, :].T[0])
                later_otherneuron_spikes_index = (self.raw_data.T[0] > i) * others_index
                first_later_otherneuron_spike_time = np.min(self.raw_data[later_otherneuron_spikes_index, :].T[0])
                gap = first_later_otherneuron_spike_time - last_earlier_otherneuron_spike_time
                if gap > 0.9 * delay_sync:
                    break
            else:
                break
        br_stop_temp = np.max(self.raw_data.T[0] * others_index)
        end_sync_index = (self.raw_data.T[0] > br_stop_temp) * sync_index
        br_stop = np.min(self.raw_data[end_sync_index, :].T[0])
        reported_duration = br_stop-sync_start
        scaling_factor = reported_duration/duration
        br_start = sync_start + delay_sync * scaling_factor
        out.t_start = br_start
        out.t_stop = br_stop
        this_f = out.firing_rate((reported_duration-delay_sync * scaling_factor)/self.y_dim)
        self.state = 1*(this_f > 0)
        #off broadcast
        matrix_b[:] = 0
        self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
        self.setup.mapper._program_onchip_broadcast_learning(matrix_b)
        if debug:
            return stim, out, self.state
        return


    def get_br(self, delta=50, n_spikes=5, neu_sync = 5, freq_sync = 400, duration_sync = 500, delay_sync = 700, debug=False):
        '''
        read synaptic matric with broadcast address
        '''
        if(delay_sync < duration_sync):
            print 'duration sync has to be < than delay_sync'
            return
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
        nindex_neu = self.pop_broadcast.synapses['virtual_exc'].addr['neu'] == (neu_sync+2)
        nsyn_sync = self.pop_broadcast.synapses['virtual_exc'][nindex_neu]
        nync_spikes = nsyn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
        #merge sync train and broadcast train
        stimulus = pyNCS.pyST.merge_sequencers(sync_spikes, stim)
        #add second sync at the end
        time_stop = np.max(stimulus[1].raw_data()[:,0]) + delay_sync
        index_neu = self.pop_broadcast.synapses['virtual_exc'].addr['neu'] == (neu_sync)
        endsync_sync = self.pop_broadcast.synapses['virtual_exc'][index_neu]
        endsync_spikes = endsync_sync.spiketrains_regular(freq_sync,duration=duration_sync, t_start=time_stop)
        stimulus = pyNCS.pyST.merge_sequencers(stimulus,endsync_spikes)
        #stimulate
        time.sleep(1.2)
        #self.setup.stimulate(sync_spikes,duration=500+500,send_reset_event=True)
        out = self.setup.stimulate(nync_spikes,send_reset_event=True)
        out = self.setup.stimulate(nync_spikes,send_reset_event=False) #should be nice to know why...
        out = self.setup.stimulate(stimulus,send_reset_event=False,duration=duration+delay_sync+duration_sync+500)
        time.sleep(1.2)
        out = out[somach]
        #clean data using no info on absolute timing
        raw_data = out.raw_data()
        sync_index = raw_data[:,1] == neu_sync

        #fix timing issue
        expected_time = delay_sync+(delta*self.y_dim)+delay_sync+duration_sync
        real_time = np.max(raw_data[sync_index,0]) - np.min(raw_data[sync_index,0])
        scaling_factor =  real_time / expected_time 
        br_start = np.min(raw_data[sync_index,0]) + delay_sync * scaling_factor 
        br_stop = np.max(raw_data[sync_index,0])  - (delay_sync + duration_sync) * scaling_factor
                
        #- delay_sync - duration_sync #br_start + (delta*self.y_dim)
        out.t_start = br_start 
        out.t_stop = br_stop
        this_f = out.firing_rate(delta) 
        self.state = 1*(this_f > 0)
        #off broadcast
        matrix_b[:] = 0
        self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
        self.setup.mapper._program_onchip_broadcast_learning(matrix_b)
        if debug:
            return stim, out, self.state
        return 

    def get(self, delta=50, isi=2, n_spikes=1, read_num=1, debug=False):
        """
        read_num = is the number of divisions in which it is intended to download the synaptic matrix 
        """
        from copy import copy as copycopy
        rates = 1./delta
        duration = delta
        somach = self.neurons.soma.channel
        sns = self.neurons.synapses[self.syntype]
        snsoma = self.neurons.soma
        # create a stimulus with 1 spike per synapse, separated by delta, for
        # all the neurons in parallel
        wij = np.zeros((self.x_dim, self.y_dim))
        from copy import copy as copycopy
        if(delta==50 and isi==2 and n_spikes==1 and read_num==1):
            load_stim = True
        else:
            load_stim = False
        if(load_stim==False):
            spiketimes = np.row_stack([np.concatenate([\
                       np.arange(i * delta, i * delta + n_spikes * isi, isi)\
                        for i in range(self.x_dim)]).reshape(-1, n_spikes)\
                                   for i in range(self.y_dim/read_num)])
        # we create bursts of 5 spikes @1kHz spaced by delta
        syn_per_read = self.y_dim/read_num
        syn_per_read_tot = (self.y_dim*self.x_dim)/read_num
        for this_read in range(read_num): 
            if(load_stim==False):
                this_syn_index = np.logical_and(sns.addr['syntype'] >= syn_per_read*this_read, sns.addr['syntype'] < syn_per_read*(this_read+1))
            # ones we have the spiketimes per cell we have to multiplex them because
            # SpikeList accepts a long list of (id, spiketime)...
                sl = r_[[zip(repeat(a, len(s)), s)\
                    for a, s in zip(sns.laddr[this_syn_index], spiketimes)]].reshape(-1, 2)
                id_list = sns.laddr[this_syn_index]
                stim = SpikeList(sl, id_list=id_list)
            else:    
                #with open('stimuli/delta50-isi2-n_spikes1-read_num1.txt', 'wb') as output: pickle.dump(stim, output, pickle.HIGHEST_PROTOCOL)
                with open('../api/wij/stimuli/delta50-isi2-n_spikes1-read_num1.txt', 'rb') as input:
                        stim = pickle.load(input)
            while True:
                try:
                    out = self.setup.stimulate({1:stim},send_reset_event=False,duration=delta*self.y_dim/read_num)
                    out = out[somach]
                    #get right edges
                    all_hist, edges = np.histogram(out.raw_data()[:,0],self.y_dim)
                    if len(out.id_list())>0:
                        laddr = snsoma.laddr
                        out.complete(laddr)
                        out = out.id_slice(laddr)
                        # now infers from the output spikes the state of the
                        # corresponding synapse
                        #plot(bins, np.linspace(10,10,len(bins)),'x')
                        for i, o in enumerate(out):
                            #bins = (self.y_dim/read_num)
                            index_to_move = np.diff(np.sort(o.spike_times)) > 0.6
                            wij[i,syn_per_read*this_read:syn_per_read*(this_read+1)] = 1. * (np.histogram(o.spike_times,edges)[0] > 0. )
                    break
                except Exception:
                    print "An error occured while acquiring. Trying again..."
                    break
            self.state = wij
        if debug:
            return stim, out, self.state
        return self.state

    def set(self, matrix, check=False):
        """
        set learning matrix
        """
        self.setup.mapper._program_onchip_learning_state(matrix)
        if(check == True):
            matrix_get = self.get()
            if(np.sum(matrix_get - matrix) != 0):
                print "error in setting matrix"

    #def set(self, matrix, check=False):
        #"""
        #"""
        #ch = self.neurons.synapses['excitatory0'].channel
        ## drive neuron 0
        #stim0 =\
            #self.neurons.synapses['excitatory0'][:1].spiketrains_regular(100,
                                                                         #duration=200)
        ## stimulate only synapses to be set to 1
        ## WARNING: check the order of stims per synapse
        #stim1 =\
            #self.neurons.synapses['learning'].spiketrains_regular(matrix*200,
                                                              #duration=200)
        ## merge
        #stim0[ch].merge(stim1[ch])
        ## configure
        #self.chip.loadBiases('biases/wij_set_allmux.biases')
        #self.reset(0)
        ## stimulate
        #out = self.setup.stimulate(stim0)
        ## check
        #if check:
            #self.get()
            #overlap = dot(self.state.flatten(), matrix.flatten())
            #print "%d/%d synapses are high, %d of which are correctly high" %\
                #(sum(self.state), sum(matrix), overlap)
            #return overlap
        #return stim0

    #def reset(self, state, sleep=2):
        #"""
        #Resets all learning weights to a given state:
            #- 0 => all weights to low
            #- 1 => all weights to high

        #Arguments:

            #sleep
                #time to wait in second between setting threshold biases
        #"""
        #oldl = self.chip.bias.psynaerlk.v
        #old = self.chip.bias.synaerth.v
        #self.chip.bias.psynaerlk.v = 2.6
        #if state:
            #self.chip.bias.synaerth.v = 0
            #time.sleep(sleep)
            #self.state =\
                    #np.ones((len(self.neurons),len(self.neurons[:1].synapses['learning'])))
        #else:
            #self.chip.bias.synaerth.v = 3.3
            #time.sleep(sleep)
            #self.state =\
                    #np.zeros((len(self.neurons),len(self.neurons[:1].synapses['learning'])))
        #self.chip.bias.psynaerlk.v = oldl
        #self.chip.bias.synaerth.v = old

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.state, cmap=cm.gray, interpolation='nearest',
                  aspect='auto', origin='bottom left')
        plt.show()
        plt.ylim([0,256])


    def load_freeme(self):
        import PIL
        from PIL import Image

        basewidth = 300

        #img = Image.open('free_me.jpg')

        im = np.asarray(Image.open('free_me.png').resize((256,256)).convert('L'))
        index_g = np.where(im > 125)
        index_l = np.where(im <= 125)
        image = np.copy(im)
        image[index_g] = 0
        image[index_l] = 1
        #self.setup.mapper._program_onchip_learning_state(image)
        return image

    def load_flower(self):
        import PIL
        from PIL import Image

        basewidth = 300

        img = Image.open('flower.jpg')

        im = np.asarray(Image.open('flower.jpg').resize((256,256)).convert('L'))
        index_g = np.where(im > 125)
        index_l = np.where(im <= 125)
        image = np.copy(im)
        image[index_g] = 0
        image[index_l] = 1
        #self.setup.mapper._program_onchip_learning_state(image)
        return image

    #def _random_matrix(self, p):
        #"""
        #Useful function for creating a new random matrix of 1/0 weights.
        #Arguments::

            #p
                #p <=> p(Wij=1)
        #"""
        #return (random(self.state.shape)<=p)*1.

    #def _random_matrix_inhomogeneous(self, p_array):
        #"""
        #Useful function for creating a new random matrix of 1/0 weights. An
        #array of [positions, probabilities] pairs can be given for an
        #inhomogeneous probability distribution of 1 values.
        #"""
        #wij = zeros_like(self.state)
        #for pos, prob in p_array:
            #wij[pos]=(random()<=prob)*1.
        #return wij

    #def _random_matrix_from_pattern(self, prototype, overlap):
        #""" 
        #Useful function for creating a new random matrix of 1/0 weights. The
        #matrix will have the given overlap with the given pattern of 0/1
        #values. Overlap must be in the range [0, 1].
        #"""
        #f = int(sum(prototype))
        #o = int(overlap*f)
        #a = permutation(argwhere(prototype==1).flatten())
        #b = permutation(argwhere(prototype==0).flatten())
        #w = concatenate([a[:o], b[:f-o]])
        #pattern = np.zeros(len(prototype))
        #pattern[w] = 1.
        #return pattern
