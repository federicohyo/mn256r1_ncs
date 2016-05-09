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

class Lsm:
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
        # end resources
        # resources for Reservoir Computing
        self.CovMatrix  = {"input":np.zeros(Nn2),"output":np.zeros(Nn2)} # Covariance matrix of inputs and outputs
        self.ReadoutW   = {"input":np.zeros([self.Nn,1]),"output":np.zeros([self.Nn,1])}     # Readout weights
        self.ProjTeach  = {"input":np.zeros([self.Nn,1]),"output":np.zeros([self.Nn,1])}     # Teaching signal projected on inputs and outputs
        alpha = np.logspace (-6,3,50) # Regularization parameters: 50 values
        self._regressor = {"input":RidgeCV(alphas=alpha,normalize=False, fit_intercept=True), \
                           "output":RidgeCV(alphas=alpha,normalize=False, fit_intercept=True)} # Linear regression with cross-validation
        self.runningMean = {"input": 0, "output":0}
        self.samples = 0
        # end resources for RC
        # network parameters
        self.cee = cee
        self.cii = cii
        self.rcn = population
        #parallel processing
        self.p = mpi.Pool(processes=4)
        if population:
            self.setup = population.setup
            self.setup.chips['mn256r1'].load_parameters('biases/biases_default.biases')
            self._init_lsm()
            self.program_config()
            self.setup.chips['mn256r1'].load_parameters('biases/biases_reservoir.biases')

    ### ========================= functions ===================================
    def _init_lsm(self):
        # rcn with learning synapses
        self._connect_populations_learning(self.rcn,self.rcn,self.cee,1)
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

    def create_spiketrain_from_matrix(self, M, c = 0.1,  max_freq= 1000, min_freq = 350, neu_sync=10, duration = 1000, delay_sync = 500, duration_sync = 200, freq_sync = 600):
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

        return stimulus

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

    def save_config(self, directory = 'lsm/'):
        '''
            save matrices configurations
        '''
        np.savetxt(directory+'conf_matrix_learning_rec.txt', self.matrix_learning_rec)
        np.savetxt(directory+'conf_matrix_learning_pot.txt', self.matrix_learning_pot)
        np.savetxt(directory+'conf_matrix_programmable_rec.txt', self.matrix_programmable_rec)
        np.savetxt(directory+'conf_matrix_programmable_w.txt', self.matrix_programmable_w)
        np.savetxt(directory+'conf_matrix_matrix_programmable_exc_inh.txt', self.matrix_programmable_exc_inh)

    def program_config(self):
        '''
        upload configuration matrices on the neuromorphic chip mn256r1
        '''
        self.setup.mapper._program_onchip_weight_matrix_programmable(self.matrix_programmable_w)
        self.setup.mapper._program_onchip_programmable_connections(self.matrix_programmable_rec)
        self.setup.mapper._program_onchip_exc_inh(self.matrix_programmable_exc_inh)
        self.setup.mapper._program_onchip_learning_state(self.matrix_learning_pot)
        self.setup.mapper._program_onchip_plastic_connections(self.matrix_learning_rec)
      
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

    def RC_poke(self, stimulus):
        '''
        c -> random connectivity from stimuli to reservoir
        nsteps -> timesteps
        num_gestures -> stuff to classify generated
        ntrials -> number of trials per gesture
        '''

        inputs, outputs = self.stimulate_reservoir(stimulus,trials=1)
        time.sleep(0.5)

        return inputs, outputs
            

    def RC_reset (self):
        self.CovMatrix  = {"input":np.zeros([self.Nn,self.Nn]),"output":np.zeros([self.Nn,self.Nn])}
        self.ReadoutW   = {"input":np.zeros([self.Nn,1]),"output":np.zeros([self.Nn,1])}
        self.ProjTeach  = {"input":np.zeros([self.Nn,1]),"output":np.zeros([self.Nn,1])}
        alpha = np.logspace (-6,3,50) #search 50 values
        self._regressor = {"input":RidgeCV(alphas=alpha,normalize=False, fit_intercept=True), \
                           "output":RidgeCV(alphas=alpha,normalize=False, fit_intercept=True)} # Linear regression with cross-validation
        self.runningMean = {"input": 0, "output":0}
        self.samples = 0
        print "RC storage reseted!"

    def RC_predict (self,x,y):
        Z = {"input":  self._regressor["input"].predict(x), \
             "output": self._regressor["output"].predict(y)}
        return Z

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
    nid = np.unique(n_id)
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
    

'''        if(save_data):

                np.savetxt("lsm/inputs_gesture_"+str(ind)+"_trial_"+str(trial)+".txt", inputs[0])
                np.savetxt("lsm/outputs_gesture_"+str(ind)+"_trial_"+str(trial)+".txt", outputs[0])

            if(learn_real_time == True):

                ac = np.mean(func_avg(self.timev[:,None], inputs[0][:,0][None,:]), axis=1) 
                ac = ac / np.max(ac)
                ac = ac[:,None]

                # Convert input and output spikes to analog signals
                #pdb.set_trace()
                X = self._ts2sig(inputs[0][:,0], np.floor(inputs[0][:,1]))
                Y = self._ts2sig(outputs[0][:,0], outputs[0][:,1])
                teach_sig = self.teach_generator(X)[:,None] * ac
               
                self.currentAnalogIn = X
                self.currentAnalogOut = Y
                self.currentTeach = teach_sig
                self._realtime_learn (X,Y,teach_sig)
                rme =[self._regressor["input"].score(X,teach_sig), self._regressor["output"].score(Y,teach_sig)]
                self.rme.append(rme)
                print "score in - out", rme
                figure(figure_rms.number)
                plot(rme,'o')

            if(do_plot_svd ==True and learn_real_time == False):

                X = self._ts2sig(inputs[0][:,0], np.floor(inputs[0][:,1]))
                Y = self._ts2sig(outputs[0][:,0], outputs[0][:,1])

            if(do_plot_svd == True):
                ac=np.mean(Y**2,axis=0)
                max_pos = np.where(ac == np.max(ac))[0]
                subplot(3,1,1)
                plot(X[:,125])
                subplot(3,1,2)
                plot(Y[:,max_pos])
                subplot(3,1,3)
                CO = np.dot(Y.T,Y)
                CI = np.dot(X.T,X)
                si = np.linalg.svd(CI, full_matrices=True, compute_uv=False)
                so = np.linalg.svd(CO, full_matrices=True, compute_uv=False)
                semilogy(so/so[0], 'bo-', label="outputs")
                semilogy(si/si[0], 'go-', label="inputs")
                legend(loc="best")
'''

