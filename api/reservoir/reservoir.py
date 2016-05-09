'''
 Copyright (C) 2014 - Federico Corradi
 Copyright (C) 2014 - Juan Pablo Carbajal
 
 This progrm is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
'''


############### author ##########
# federico corradi
# federico@ini.phys.ethz.ch
# Juan Pablo Carbajal 
# ajuanpi+dev@gmail.com
#
# Liquid State Machine class mn256r1 
# ===============================
from __future__ import division

### ========================= import packages ===============================
import random
import numpy as np
import time
import pyNCS
import matplotlib
from pylab import *
from sklearn.linear_model import RidgeCV
from sklearn import metrics
from scipy.stats import pearsonr

import pdb
import multiprocessing as mpi

class Reservoir:
    def __init__(self, population=None,  cee=0.5, cii=0.3,nx=16,ny=16):
        ### ========================= define what is needed to program the chip ====
        self.shape = (nx,ny);
        self.Nn    = np.prod(self.shape);
        Nn2 = [self.Nn,self.Nn]
        # resources
        self.matrix_learning_rec = np.zeros(Nn2)
        self.matrix_learning_pot = np.zeros(Nn2)
        self.matrix_programmable_rec = np.zeros(Nn2)
        self.matrix_programmable_w = np.zeros(Nn2)
        self.matrix_programmable_exc_inh = np.zeros(Nn2)
        self.matrix_broadcast = np.ones(Nn2)
        # end resources
        # resources for Reservoir Computing
        self.CovMatrix  = {"input":np.zeros(Nn2),"output":np.zeros(Nn2)} # Covariance matrix of inputs and outputs
        self.ReadoutW   = {"input":np.zeros([self.Nn,1]),"output":np.zeros([self.Nn,1])}     # Readout weights
        self.ProjTeach  = {"input":np.zeros([self.Nn,1]),"output":np.zeros([self.Nn,1])}     # Teaching signal projected on inputs and outputs
        self.alpha = np.logspace (0.1,50,1) # Regularization parameters: 50 values
        self._regressor = {"input":RidgeCV(alphas=self.alpha,normalize=True, fit_intercept=True), \
                           "output":RidgeCV(alphas=self.alpha,normalize=True, fit_intercept=True)} # Linear regression with cross-validation
        self.runningMean = {"input": 0, "output":0}
        self.samples = 0
        # end resources for RC
        # network parameters
        self.cee = cee
        self.cii = cii
        self.rcn = population
        if population:
            self.setup = population.setup
            #self.setup.chips['mn256r1'].load_parameters('biases/biases_default.biases')
            self._init_lsm()
            self.program_config()
            #self.setup.chips['mn256r1'].load_parameters('biases/biases_reservoir.biases')

    ### ========================= functions ===================================
    def _init_lsm(self):
        # rcn with learning synapses
        #self._connect_populations_learning(self.rcn,self.rcn,self.cee,1)
        self._connect_populations_programmable(self.rcn,self.rcn,self.cee,[0,3])
        self._connect_populations_programmable_inh(self.rcn,self.rcn,self.cii,[0,3])
        return 

    def _connect_populations_programmable_inh(self, pop_pre,pop_post,connectivity,w):
        '''
            Connect two populations via programmable synapses with specified connectivity and w
        '''    
        if(np.shape(w)[0] == 2):
            w_min = w[0]        
            w_max = w[1]
            random_w = True
        elif(np.shape(w)[0] == 1 ):
            random_w = False
        else:
            print 'w should be np.shape(2), [w_min,w_max]'

        #loop trought the pop and connect with probability connectivity
        for pre in pop_pre.soma.addr['neu']:
            for post in pop_post.soma.addr['neu']:
                coin = np.random.rand()
                if(coin < connectivity):
                    #we connect this pre with this post
                    self.matrix_programmable_exc_inh[post,pre] = 0  
                    self.matrix_programmable_rec[post,pre] = 1    
                    if(random_w):
                        self.matrix_programmable_w[post,pre] = np.random.randint(w_max)+w_min
                    else:
                        self.matrix_programmable_w[post,pre] = w[0]

    def _connect_populations_learning(self, pop_pre,pop_post,connectivity,pot):
        '''
            Connect two populations via learning synapses with specified connectivity and pot
        '''    
        #loop trought the pop and connect with probability connectivity
        for pre in pop_pre.soma.addr['neu']:
            for post in pop_post.soma.addr['neu']:
                coin = np.random.rand()
                if(coin < connectivity):
                    #we connect this pre with this post
                    self.matrix_learning_rec[post,pre] = 1 
                    coin = np.random.rand()
                if(coin < pot):  
                    self.matrix_learning_pot[post,pre] = 1

    def _connect_populations_programmable(self, pop_pre, pop_post,connectivity,w):
        '''
            Connect two populations via programmable synapses with specified connectivity and w
        '''    
        if(np.shape(w)[0] == 2):
            w_min = w[0]        
            w_max = w[1]
            random_w = True
        elif(np.shape(w)[0] == 1 ):
            random_w = False
        else:
            print 'w should be np.shape(2), [w_min,w_max]'

        #loop trought the pop and connect with probability connectivity
        for pre in pop_pre.soma.addr['neu']:
            for post in pop_post.soma.addr['neu']:
                coin = np.random.rand()
                if(coin < connectivity):
                    #we connect this pre with this post
                    self.matrix_programmable_exc_inh[post,pre] = 1
                    self.matrix_programmable_rec[post,pre] = 1   
                    if(random_w):
                        self.matrix_programmable_w[post,pre] = np.random.randint(w_max)+w_min
                    else:
                        self.matrix_programmable_w[post,pre] = w[0]

    def create_stimuli_matrix (self, G, rates, nT, nx=16, ny=16) :
        '''
        Generates a matrix the mean rates of the input neurons defined in
        nT time intervals.
        
        ** Inputs **
        G: A list with G_i(x,y), each element is a function 
                G_i: [-1,1]x[-1,1] --> [0,1] 
           defining the intensity of mean rates on the (-1,1)-square.
        nx,ny: Number of neurons in the x and y direction (default 16).

        rates: A list with the time variations of the input mean rates.
               Each element is
                f_i: [0,1] --> [0,1]
        
        nT: Number of time intervals.

        ** Outputs **
        '''

        nR = len(rates) # Number of rates

        x,y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny))
        t   = np.linspace(0,1,nT,endpoint=False)
        V   = np.array([r(t) for r in rates])

        M = np.zeros ([nx*ny, nT])
        GG = np.zeros ([nx,ny])
        for g,r in zip(G,V):
            M += np.array(g(x,y).ravel()[:,None] * r) / sum (g(x,y).ravel()[:,None])
            GG += g(x,y)

        return M

    def _qmerge(self, l):
        '''
        helper function used to merge spiketrains
        '''
        if len(l) == 1:
            return l
        elif len(l) == 2:
            return [pyNCS.pyST.merge_sequencers(l[0],l[1])]
        else:
            a = self._qmerge(l[:int(len(l)/2)])
            b = self._qmerge(l[int(len(l)/2):])
            a.extend(b)
        return a

    def merge_spiketrain_list(self, L):
        '''
        daje con il merge
        '''
        x = self._qmerge(L)
        while len(x) > 1:
            x = self._qmerge(x)
        ret = x[0]
        return ret

    def create_spiketrain_from_matrix_reg_fast(self, M, c = 0.1,  max_freq= 1000, min_freq = 350, neu_sync=10, duration = 1000, delay_sync = 500, duration_sync = 200, freq_sync = 600, index_syn=None):
        '''
        create stimulus from rate matrix
        it adds the sync neu as well, it does not use inhibitory poisson stimuli         
        '''
        vsyn = 4
        somach = self.rcn.soma.channel
        inputch = 1
        nneu, nsteps = np.shape(M)
        time_bin = duration/float(nsteps)

        #we pick a random projection
        nsyn_tot = len(self.rcn.synapses['virtual_exc'].addr)
        syn_for_input_neu = int(c*nsyn_tot)
        if(index_syn == None):
            index_syn = np.random.random_integers(0, high=nsyn_tot-1, size=nneu*syn_for_input_neu)

        #build up stimulus
        timebins = np.linspace(0, duration, nsteps)
        #syn = self.rcn.synapses['virtual_exc'][index_syn.astype(int)]
        #rescale M to max/min freq
        new_value = np.ceil(( (M - np.min(M)) / (np.max(M) - np.min(M)+1e-12) ) * (max_freq  - min_freq) + min_freq)
        
        #sync stimulus 
        index_neu = self.rcn.synapses['virtual_exc'].addr['neu'] == neu_sync           
        syn_sync = self.rcn.synapses['virtual_exc'][index_neu]
        sync_spikes = syn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
        #for every neuson create stimulus projection
        syn = self.rcn.synapses['virtual_exc'][0]
        spiketrain = syn.spiketrains_regular(0, duration=10)
        stimulus_s = pyNCS.pyST.merge_sequencers(sync_spikes, spiketrain)
        final_seqs = []
        
        this_tot_M = []
        syn_tot_M = []
        
        syn = self.rcn.synapses['virtual_exc'][index_syn.astype(int)]
        this_M = np.vstack((new_value,)*syn_for_input_neu)
        final_seqs = []    
        #for this_t in range(nsteps):
        #    print 'processing stimulus time_step n:', this_t
        #    this_phase_train = syn.spiketrains_regular(this_M[:,this_t], t_start = this_t*time_bin+delay_sync,  duration = time_bin, jitter=False)
        #    final_seqs.append(this_phase_train)
        #to_use = int(np.floor(np.sqrt(len(final_seqs))))
        #stimulus = self.merge_spiketrain_list(final_seqs[0:to_use**2-1])
        #stimulus = pyNCS.pyST.merge_sequencers(stimulus,stimulus_s)
        
        #SpikeList accepts a long list of (id, spiketime)
        tot_stim = []
        sl = np.array([0,0])
        for this_t in range(nsteps):
            n_spikes = np.floor(this_M[:,this_t]/(time_bin))
            spiketimes = [ np.linspace(this_t*time_bin+delay_sync,(this_t+1)*time_bin+delay_sync,n_spikes[i]) for i in range(len(n_spikes))]
            sl = np.vstack([zip(repeat(a, len(s)), s) for a, s in zip(syn.laddr, spiketimes)])
            id_list = syn.laddr 
            stim = pyNCS.pyST.SpikeList(sl, id_list=id_list)
            tot_stim.append({1:stim})
        to_use = int(np.floor(np.sqrt(len(tot_stim))))
        stimulus = self.merge_spiketrain_list(tot_stim[0:to_use**2-1])
        stimulus = pyNCS.pyST.merge_sequencers(stimulus,stimulus_s)
        
        return stimulus, index_syn                                    

        #spiketimes = np.row_stack([np.concatenate([np.arange(i * 50, i * 50 + 5 * 1, 1) for i in range(256)]).reshape(-1, 5) for i in range(256)])

    def create_spiketrain_from_amplitude(self, M, c = 0.1,  max_freq= 1000, min_freq = 350, neu_sync=10, duration = 1000, delay_sync = 500, duration_sync = 200, freq_sync = 600, index_syn=None):
        '''
        create stimulus from rate matrix
        it adds the sync neu as well, it does not use inhibitory poisson stimuli         
        '''
        vsyn = 4
        somach = self.rcn.soma.channel
        inputch = 1
        nneu, nsteps = np.shape(M)
        time_bin = duration/float(nsteps)

        #we pick a random projection
        nsyn_tot = len(self.rcn.synapses['virtual_exc'].addr)
        syn_for_input_neu = int(c*nsyn_tot)
        if(index_syn == None):
            index_syn = np.random.random_integers(0, high=nsyn_tot-1, size=nneu*syn_for_input_neu)

        #build up stimulus
        timebins = np.linspace(0, duration, nsteps)
        #syn = self.rcn.synapses['virtual_exc'][index_syn.astype(int)]
        #rescale M to max/min freq
        new_value = np.ceil(( (M - 0) / (1.0 - 0) ) * (max_freq  - min_freq) + min_freq)
        
        #sync stimulus 
        index_neu = self.rcn.synapses['virtual_exc'].addr['neu'] == neu_sync           
        syn_sync = self.rcn.synapses['virtual_exc'][index_neu]
        sync_spikes = syn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
        #for every neuson create stimulus projection
        syn = self.rcn.synapses['virtual_exc'][0]
        spiketrain = syn.spiketrains_regular(0, duration=10)
        stimulus_s = pyNCS.pyST.merge_sequencers(sync_spikes, spiketrain)
        final_seqs = []
        
        this_tot_M = []
        syn_tot_M = []
        
        syn = self.rcn.synapses['virtual_exc'][index_syn.astype(int)]
        this_M = np.vstack((new_value,)*syn_for_input_neu)
        final_seqs = []    
        #SpikeList accepts a long list of (id, spiketime)
        tot_stim = []
        sl = np.array([0,0])
        for this_t in range(nsteps):
            n_spikes = np.floor(this_M[:,this_t]/(time_bin))
            spiketimes = [ np.linspace(this_t*time_bin+delay_sync,(this_t+1)*time_bin+delay_sync,n_spikes[i]) for i in range(len(n_spikes))]
            sl = np.vstack([zip(repeat(a, len(s)), s) for a, s in zip(syn.laddr, spiketimes)])
            id_list = syn.laddr 
            stim = pyNCS.pyST.SpikeList(sl, id_list=id_list)
            tot_stim.append({1:stim})
        to_use = int(np.floor(np.sqrt(len(tot_stim))))
        stimulus = self.merge_spiketrain_list(tot_stim[0:to_use**2-1])
        stimulus = pyNCS.pyST.merge_sequencers(stimulus,stimulus_s)
        
        return stimulus, index_syn     
                                       
    def create_spiketrain_from_matrix_reg(self, M, c = 0.1,  max_freq= 1000, min_freq = 350, neu_sync=10, duration = 1000, delay_sync = 500, duration_sync = 200, freq_sync = 600):
        '''
        create stimulus from rate matrix
        it adds the sync neu as well, it does not use inhibitory poisson stimuli
        '''
        vsyn = 4
        somach = self.rcn.soma.channel
        inputch = 1
        nneu, nsteps = np.shape(M)
        time_bin = duration/float(nsteps)

        #we pick a random projection
        nsyn_tot = len(self.rcn.synapses['virtual_exc'].addr)
        syn_for_input_neu = int(c*nsyn_tot)
        index_syn = np.random.random_integers(0, high=nsyn_tot-1, size=nneu*syn_for_input_neu)

        #build up stimulus
        timebins = np.linspace(0, duration, nsteps)
        #syn = self.rcn.synapses['virtual_exc'][index_syn.astype(int)]
        #rescale M to max/min freq
        new_value = np.ceil(( (M - np.min(M)) / (np.max(M) - np.min(M)) ) * (max_freq  - min_freq) + min_freq)
        
        #sync stimulus 
        index_neu = self.rcn.synapses['virtual_exc'].addr['neu'] == neu_sync           
        syn_sync = self.rcn.synapses['virtual_exc'][index_neu]
        sync_spikes = syn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
        #for every neuson create stimulus projection
        syn = self.rcn.synapses['virtual_exc'][0]
        spiketrain = syn.spiketrains_regular(0, duration=10)
        stimulus_s = pyNCS.pyST.merge_sequencers(sync_spikes, spiketrain)
        final_seqs = []
        for this_neu in range(nneu):
            print 'processing stimulus neuron n:', this_neu
            syn = self.rcn.synapses['virtual_exc'][index_syn.astype(int)][this_neu*syn_for_input_neu:(this_neu+1)*syn_for_input_neu]
            this_M = np.vstack((new_value[this_neu,:],)*syn_for_input_neu)
            for this_t in range(nsteps):
                this_phase_train = syn.spiketrains_regular(this_M[:,this_t], t_start = this_t*time_bin+delay_sync,  duration = time_bin)    
                final_seqs.append(this_phase_train) 
        to_use = int(np.floor(np.sqrt(len(final_seqs))))
        stimulus = self.merge_spiketrain_list(final_seqs[0:to_use**2-1])
        stimulus = pyNCS.pyST.merge_sequencers(stimulus,stimulus_s)

        return stimulus

    def create_spiketrain_from_matrix(self, M, c = 0.1,  max_freq= 1000, min_freq = 350, neu_sync=10, duration = 1000, delay_sync = 500, duration_sync = 200, freq_sync = 600, index_syn=None):
        '''
        create stimulus from rate matrix 
        it adds the sync neu as well
        '''
        vsyn = 4
        somach = self.rcn.soma.channel
        inputch = 1
        nneu, nsteps = np.shape(M)

        #we pick a random projection
        nsyn_tot = len(self.rcn.synapses['virtual_exc'].addr)
        syn_for_input_neu = int(c*nsyn_tot)
        if(index_syn == None):
            index_syn = np.random.random_integers(0, high=nsyn_tot-1, size=nneu*syn_for_input_neu)


        #build up stimulus
        timebins = np.linspace(0, duration, nsteps)
        #syn = self.rcn.synapses['virtual_exc'][index_syn.astype(int)]
        #rescale M to max/min freq
        new_value = np.ceil(( (M - np.min(M)) / (np.max(M) - np.min(M)) ) * (max_freq  - min_freq) + min_freq)
        
        #sync stimulus 
        index_neu = self.rcn.synapses['virtual_exc'].addr['neu'] == neu_sync           
        syn_sync = self.rcn.synapses['virtual_exc'][index_neu]
        sync_spikes = syn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
        #for every neuson create stimulus projection
        syn = self.rcn.synapses['virtual_exc'][0]
        spiketrain = syn.spiketrains_regular(0, duration=10)
        stimulus = pyNCS.pyST.merge_sequencers(sync_spikes, spiketrain)
        for this_neu in range(nneu):
            syn = self.rcn.synapses['virtual_exc'][index_syn.astype(int)][this_neu*syn_for_input_neu:(this_neu+1)*syn_for_input_neu]
            this_M = np.vstack((new_value[this_neu,:],)*syn_for_input_neu)
            spiketrain = syn.spiketrains_inh_poisson(this_M,timebins+delay_sync)
            stimulus = pyNCS.pyST.merge_sequencers(stimulus, spiketrain)

        return stimulus, index_syn

    def stimulate_reservoir(self, stimulus, neu_sync = 10,  trials=5):
        '''
        stimulate reservoir via virtual input synapses
        nsteps -> time steps to be considered in a duration = duration
        max_freq -> max input freq
        min_freq -> min input freq
        trials -> number of different stimulations with inhonogeneous poisson spike trains
        rate_matrix -> normalized rate matrix with dimensions [nsyn,timebins]
        '''
        duration = np.max(stimulus[1].raw_data()[:,0])-np.min(stimulus[1].raw_data()[:,0])+1000
        somach = self.rcn.soma.channel
        tot_outputs = []
        tot_inputs = []
        for this_stim in range(trials):
            out = self.setup.stimulate(stimulus, send_reset_event=False, duration=duration)
            out = out[somach]
            #sync data with sync neuron
            raw_data = out.raw_data()
            sync_index = raw_data[:,1] == neu_sync
            start_time = np.min(raw_data[sync_index,0])
            index_after_sync = raw_data[:,0] > start_time
            clean_data = raw_data[index_after_sync,:]
            clean_data[:,0] = clean_data[:,0]-np.min(clean_data[:,0])
            #copy synched data
            tot_outputs.append(clean_data)
            tot_inputs.append(stimulus[1].raw_data())
            #if(plot_samples == True):
            #    Y = self._ts2sig(tot_outputs[this_stim][:,0], tot_outputs[this_stim][:,1])
            #    for i in range(256):
            #        figure(figure_h.number)
            #        subplot(16,16,i)
            #        plot(Y[:,i])

        return tot_inputs, tot_outputs

    def plot_inout(self, inputs,outputs):
        '''
        just plot input and output trials of reservoir after stimulate reservoir
        '''
        trials = len(inputs)
        figure()
        for i in range(trials):
            plot(inputs[i][:,0],inputs[i][:,1],'o')
            xlabel('time [s]')
            ylabel('neuron id')
            title('input spike trains')
        for i in range(trials):
            figure()
            plot(outputs[i][:,0],outputs[i][:,1],'o')
            xlabel('time [s]')
            ylabel('neuron id')
            title('output spike trains')

    def load_config(self, directory='lsm/'):
        '''
            load configuration from folder 
        '''
        self.matrix_learning_rec = np.loadtxt(directory+'conf_matrix_learning_rec.txt')
        self.matrix_learning_pot = np.loadtxt(directory+'conf_matrix_learning_pot.txt')
        self.matrix_programmable_rec = np.loadtxt(directory+'conf_matrix_programmable_rec.txt')
        self.matrix_programmable_w = np.loadtxt(directory+'conf_matrix_programmable_w.txt')
        self.matrix_programmable_exc_inh = np.loadtxt(directory+'conf_matrix_matrix_programmable_exc_inh.txt')
        self.matrix_broadcast = np.loadtxt(directory+'conf_matrix_matrix_broadcast_programmable.txt')

    def save_config(self, directory = 'lsm/'):
        '''
            save matrices configurations
        '''
        np.savetxt(directory+'conf_matrix_learning_rec.txt', self.matrix_learning_rec)
        np.savetxt(directory+'conf_matrix_learning_pot.txt', self.matrix_learning_pot)
        np.savetxt(directory+'conf_matrix_programmable_rec.txt', self.matrix_programmable_rec)
        np.savetxt(directory+'conf_matrix_programmable_w.txt', self.matrix_programmable_w)
        np.savetxt(directory+'conf_matrix_matrix_programmable_exc_inh.txt', self.matrix_programmable_exc_inh)
        np.savetxt(directory+'conf_matrix_matrix_broadcast_programmable.txt', self.matrix_broadcast)

    def show_config(self, directory = 'lsm/'):
        '''
            save matrices configurations
        '''
        figure()
        subplot(3,3,1)
        title('matrix learning rec')
        imshow(self.matrix_learning_rec)
        subplot(3,3,2)
        title('matrix learning pot')
        imshow(self.matrix_learning_pot)
        subplot(3,3,3)
        title('matrix program rec')
        imshow(self.matrix_programmable_rec)
        subplot(3,3,4)
        title('matrix program w')
        imshow(self.matrix_programmable_w)
        subplot(3,3,5)
        title('matrix program e/i')
        imshow(self.matrix_programmable_exc_inh)
        subplot(3,3,6)
        title('matrix program broadcast')
        imshow(self.matrix_broadcast)

    def program_config(self):
        '''
        upload configuration matrices on the neuromorphic chip mn256r1
        '''
        self.setup.mapper._program_onchip_weight_matrix_programmable(self.matrix_programmable_w)
        self.setup.mapper._program_onchip_programmable_connections(self.matrix_programmable_rec)
        self.setup.mapper._program_onchip_exc_inh(self.matrix_programmable_exc_inh)
        self.setup.mapper._program_onchip_learning_state(self.matrix_learning_pot)
        self.setup.mapper._program_onchip_plastic_connections(self.matrix_learning_rec)
        self.setup.mapper._program_onchip_broadcast_programmable(self.matrix_broadcast)
      
    def _ismember(self, a,b):
        '''
        as matlab: ismember
        '''
        # tf = np.in1d(a,b) # for newer versions of numpy
        tf = np.array([i in b for i in a])
        u = np.unique(a[tf])
        index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
        return tf, index

    def mean_neu_firing(self, spike_train, n_neurons,nbins=10):
        '''
        return mean neu firing matrix
        '''
        simulation_time = [np.min(spike_train[:,0]),np.max(spike_train[:,0])]
        un, bins = np.histogram(simulation_time,nbins)
        mean_rate = np.zeros([len(n_neurons),nbins])
        for b in range(nbins-1):
            #simulation_time = [np.min(spike_train[0][:]), np.max(spike_train[0][:])]
            for i in range(len(n_neurons)):
                index_neu = np.where( \
                              np.logical_and(spike_train[:,1] == n_neurons[i], \
                                     np.logical_and(spike_train[:,0] > bins[b], \
                                            spike_train[:,0] < bins[b+1] ) \
                                            ) \
                                    )
                mean_rate[i,b] = len(index_neu[0])*1000.0/(bins[b+1]-bins[b]) # time unit: ms
        return mean_rate

    def poke(self, stimulus):
        '''
        c -> random connectivity from stimuli to reservoir
        nsteps -> timesteps
        num_gestures -> stuff to classify generated
        ntrials -> number of trials per gesture
        '''

        inputs, outputs = self.stimulate_reservoir(stimulus,trials=1)
        time.sleep(0.5)

        return inputs, outputs
            

    def reset (self, alpha=np.logspace (-12,40,100)):
        '''
        reset reservoir
        '''
        self.alpha = alpha
        self.CovMatrix  = {"input":np.zeros([self.Nn,self.Nn]),"output":np.zeros([self.Nn,self.Nn])} # Covariance matrix of inputs and outputs
        self.ReadoutW   = {"input":np.zeros([self.Nn,1]),"output":np.zeros([self.Nn,1])}     # Readout weights
        self.ProjTeach  = {"input":np.zeros([self.Nn,1]),"output":np.zeros([self.Nn,1])}     # Teaching signal projected on inputs and outputs
        #alpha = np.logspace (-12,40,100) # Regularization parameters: 50 values
        self._regressor = {"input":RidgeCV(alphas=self.alpha,normalize=True, fit_intercept=False), \
                           "output":RidgeCV(alphas=self.alpha,normalize=True, fit_intercept=False)} # Linear regression with cross-validation
        self.runningMean = {"input": 0, "output":0}
        self.samples = 0
        
        print "RC storage reseted!"

    def train(self, X, Yt=None, teach_sig=None):
        '''
        Regression of teach_sig using inputs (Yt) and outputs (X)
        '''
        #inits
        nT,Nn        = np.shape(X)
        nTtot        = self.samples + nT
        w            = (self.samples/nTtot, 1.0/nTtot)      
        # Covariance matrix
        Cx = np.dot (X.T, X) # output
        # Projection of data
        Zx = np.dot (X.T, teach_sig) #output
        # Update cov matrix
        #raise Exception
        self.CovMatrix["output"]  = \
                    w[0]*self.CovMatrix["output"] + w[1]*Cx
        self.ProjTeach["output"]  = \
                    w[0]*self.ProjTeach["output"] + w[1]*Zx
        # Update weights
        self._regressor["output"].fit(self.CovMatrix["output"],\
                                      self.ProjTeach["output"])
        self.ReadoutW["output"] = self._regressor["output"].coef_.T

        
        C  = np.dot (Yt.T, Yt) # input
        Z  = np.dot (Yt.T, teach_sig) # input
        self.CovMatrix["input"]  = \
                w[0]*self.CovMatrix["input"] + w[1]*C
        self.ProjTeach["input"]  = \
                w[0]*self.ProjTeach["input"] + w[1]*Z
        # Update weights
        self._regressor["input"].fit(self.CovMatrix["input"],\
                                     self.ProjTeach["input"])
        self.ReadoutW["input"]  = self._regressor["input"].coef_.T

        # Update samples
        self.samples = nTtot

    def predict (self, X, Yt=None,  initNt=0):
        '''
        X -> outputs
        Yt -> inputs
        '''
        
        Z ={"output": self._regressor["output"].predict(X[initNt::,:])}
        Z["input"] = self._regressor["input"].predict(Yt[initNt::,:])

        return Z
                           
    def create_stimuli_matrix (self, G, rates, nT, nx=16, ny=16) :
        '''
        Generates a matrix the mean rates of the input neurons defined in
        nT time intervals.
        
        ** Inputs **
        G: A list with G_i(x,y), each element is a function 
                G_i: [-1,1]x[-1,1] --> [0,1] 
           defining the intensity of mean rates on the (-1,1)-square.
        nx,ny: Number of neurons in the x and y direction (default 16).

        rates: A list with the time variations of the input mean rates.
               Each element is
                f_i: [0,1] --> [0,1]
        
        nT: Number of time intervals.

        ** Outputs **
        '''
        
        nR = len(rates) # Number of rates

        x,y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny))
        t   = np.linspace(0,1,nT,endpoint=False)
        V   = np.array([r(t) for r in rates])

        M = np.zeros ([nx*ny, nT])
        GG = np.zeros ([nx,ny])
        for g,r in zip(G,V):
            M += np.array(g(x,y).ravel()[:,None] * r) / sum (g(x,y).ravel()[:,None])
            GG += g(x,y)

        return M
        
    def generates_gestures(self, num_gestures, n_components, max_f = 8, min_f = 1,  nScales = 4):
        '''
        generates gesture with n_components in frequency
        '''
        gestures = []
        for this_gesture in range(num_gestures):
            freqs   = (np.random.randint(min_f, high=max_f,size=n_components)+1).tolist()   # in Hz
            centers = (-1+2*np.random.random((n_components,2))).tolist()
            width   = (0.5+np.random.random(n_components)).tolist()
            gestures.append({'freq': freqs, 'centers': centers, 'width': width})   
        rates = []
        G     = []
        for ind,this_g in enumerate(gestures):
          for f  in this_g['freq']:
              rates.append(lambda t,w=f: 0.5+0.5*np.sin(2*np.pi*w*t)) 
          # Multiple spatial distribution
          for width,pos in zip(this_g['width'], this_g['centers']):
              G.append(lambda x,y,d=width,w=pos: np.exp ((-(x-w[0])**2 + (y-w[1])**2)/d**2))
    
        return G, rates, gestures

    def generates_G_rates(self, gestures): 
        rates = []
        G     = []
        for ind,this_g in enumerate(gestures):
          for f  in this_g['freq']:
              rates.append(lambda t,w=f: 0.5+0.5*np.sin(2*np.pi*w*t)) 
          # Multiple spatial distribution
          for width,pos in zip(this_g['width'], this_g['centers']):
              G.append(lambda x,y,d=width,w=pos: np.exp ((-(x-w[0])**2 + (y-w[1])**2)/d**2))
        return G, rates
        
    def generate_teacher(self, gesture, rates, n_components, nT, nScales, timev, teach_scale):
        #generate teaching signal associated with the Gesture
        teach_sig = np.zeros([nT, nScales])
        for this_component in range(n_components):
            #sum all frequencies with distance dependent from centers
            this_centers = np.array(gesture['centers'])
            #rates are weighted with their respective euclidan distance from center
            rate_w = np.sqrt(this_centers[this_component,0]**2 + this_centers[this_component,1]**2)
            teach_sig += rates[this_component]((teach_scale*timev[:,None])*1e-3)*rate_w          
        return teach_sig
        
    def root_mean_square(self, ideal, measured, norm=False):
        ''' calculate RMSE 
        ie: root_mean_square(ideal,measured)
        numpy vector in 
        float out'''
        import numpy as np
        ret = np.sqrt(((ideal - measured) ** 2).mean())
        if norm:
          ret = ret / np.sqrt((ideal ** 2).mean())
        return ret

    ### HELPER FUNCTIONS
    def _realtime_learn (self, x, y, teach_sig):
        '''
        Regression of teach_sig using inputs (x) and outputs (y).
        '''
        nT,Nn        = x.shape
        nTtot        = self.samples + nT
        w            = (self.samples/nTtot, 1.0/nTtot)

        # Update mean
        #self.runningMean["input"] = w[0]*self.runningMean["input"] + w[1]*np.sum(x,axis=0)
        #self.runningMean["output"] = w[0]*self.runningMean["output"] + w[1]*np.sum(y,axis=0)
        # Detrend data
        #xx = x - self.runningMean["input"]
        #yy = y - self.runningMean["output"]
        # Covariance matrix
        Cx = np.dot (x.T, x) # input
        C  = np.dot (y.T, y) # output
        # Projection of data
        Zx = np.dot (x.T, teach_sig)
        Z  = np.dot (y.T, teach_sig)
        #print  "covdiff ", np.sum(np.abs(self.CovMatrix["input"]-Cx))/np.sum(np.abs(self.CovMatrix["input"]))
        # Update cov matrix
        self.CovMatrix["input"]   = w[0]*self.CovMatrix["input"] + w[1]*Cx
        self.CovMatrix["output"]  = w[0]*self.CovMatrix["output"] + w[1]*C
        self.ProjTeach["input"]   = w[0]*self.ProjTeach["input"] + w[1]*Zx
        self.ProjTeach["output"]  = w[0]*self.ProjTeach["output"] + w[1]*Z
        # Update weights
        self._regressor["input"].fit(self.CovMatrix["input"], self.ProjTeach["input"])
        self._regressor["output"].fit(self.CovMatrix["output"], self.ProjTeach["output"])
        self.ReadoutW["input"]  = self._regressor["input"].coef_.T
        self.ReadoutW["output"] = self._regressor["output"].coef_.T

        #self._regressor["input"].fit(x, teach_sig)
        #self._regressor["output"].fit(y, teach_sig)
        #if (self.samples == 0) :
        #    self.ReadoutW["input"]  = self._regressor["input"].coef_.T
        #    self.ReadoutW["output"] = self._regressor["output"].coef_.T
        #else:
        #    self.ReadoutW["input"]  = w[0]*self.ReadoutW["input"] + w[1]*self._regressor["input"].coef_.T
        #    self.ReadoutW["output"] = w[0]*self.ReadoutW["output"] + w[1]*self._regressor["output"].coef_.T

        # Update samples
        self.samples = nTtot

    #characterize neurons responses
    def measure_phy_neurons(self, max_freq= 1500, min_freq = 350, duration = 1000, nsteps = 5):
        '''
        ideally we would like tanh
        '''
        vsyn = 4
        somach = self.rcn.soma.channel
        inputch = 1
        neu_sync = 10
        freq_sync = 1500
        duration_sync = 100
        #save config
        self.save_config()
        #reset connection matrix
        m_z = np.zeros([256,256])
        self.setup.mapper._program_onchip_plastic_connections(m_z)
        self.setup.mapper._program_onchip_programmable_connections(m_z)
        
        #regular stimulation for neurons 
        index_neu = self.rcn.synapses['virtual_exc'].addr['neu'] == neu_sync           
        syn_sync = self.rcn.synapses['virtual_exc'][index_neu]
        tot_spike_phases = syn_sync.spiketrains_regular(freq_sync,duration=duration_sync)
        syn = self.rcn.synapses['virtual_exc']

        freqs = np.linspace(min_freq, max_freq, nsteps)
        stim_time = float(duration)/nsteps
        for i in range(nsteps):
            t_start = (stim_time*i)+duration_sync
            spiketrain = syn.spiketrains_poisson(freqs[i], duration=stim_time, t_start = (stim_time * i)+duration_sync)
            tot_spike_phases = pyNCS.pyST.merge_sequencers(spiketrain, tot_spike_phases)

        out = self.setup.stimulate(tot_spike_phases,send_reset_event=False, duration=duration*2) 
        out = out[somach]
        raw_data = out.raw_data()
        sync_index = raw_data[:,1] == neu_sync
        start_time = np.min(raw_data[sync_index,0])
        #index_after_sync = raw_data[:,0] > start_time
        #clean_data = raw_data[index_after_sync,:]
        #clean_data[:,0] = clean_data[:,0]- np.min(clean_data[:,0])

        figure()
        out.t_start = start_time
        plot(freqs[0:nsteps],np.sum(out.firing_rate(stim_time), axis=0)[0:nsteps]/256.0, 'o-')
        xlabel(r'$\nu_{in}$ [Hz]')
        ylabel(r'$\nu_{out}$ [Hz]') 

        return
        
    def RC_score(self, zh, sig):
        '''
        compare two signals
        '''
        #scores = np.sqrt(((zh - sig) ** 2).sum(axis=0)) / np.sqrt((sig** 2).sum(axis=0))
        nT, nteach = np.shape(sig)
        scores = np.zeros([nteach,2])
        for i in range(nteach):
            scores[i,:] = pearsonr(zh[:,i], sig[:,i])
            
        #nT, nteach = np.shape(sig)
        #z = (zh-zh.mean(axis=0))/zh.std(axis=0)
        #s = (sig-sig.mean(axis=0))/sig.std(axis=0)
        #scores = np.dot(z.T,s).T / nT
        
        #scores = np.zeros([nteach,1])
        #for ind,this_teach in enumerate(teach_sig.T):
        #    sig  = np.dot(X,self._regressor[type_r].coef_[ind,:][:,None])+self._regressor[type_r].intercept_[ind]
        #    sig = (sig - np.mean(sig))/np.std(sig) 
        #    tt = np.copy(this_teach)
        #    tt = (tt-np.mean(tt))/np.std(tt)
        #    score = np.dot(sig.T,tt)  / nT
        #    scores[ind,:] = score
            
        return scores

def ts2sig (t, func, ts, n_id, n_neu = 256):
    '''
    t -> time vector
    func -> time basis f(t,ts)
    ts - > time stamp of spikes
    n_id -> neuron id
    '''
    nT = len(t)
    nid = map(int,np.unique(n_id))
    nS = len(nid)
    Y = np.zeros([nT,n_neu])
    tot_exponent = []
    for i in xrange(nS):
        idx = np.where(n_id == nid[i])[0]
        #tot_exponent.append([this_exp])
        #for j in range(len(idx)):
        Y[:,nid[i]] = np.sum(func(t[:,None],ts[idx]), axis=1)
        #Y = self.p.map(np.exp, tot_exponent)
    return Y

def orth_signal (x, atol=1e-13, rtol=0):
    '''
    Returns signal orthogonal to input ensemble.
    x -> input singal [n_samples, n_neurons]
    '''
    t = np.linspace(0,1,x.shape[0])[:,None]
    f = arange(x.shape[1])/x.shape[1]
    xt = np.sum(sin(2*np.pi*f*3*t)/(f+1),axis=1)
    w = RidgeCV(np.logspace(-6,3,50))
    w.fit(x,xt)
    xt = xt - w.predict(x)
    #pdb.set_trace()
    return xt
    




