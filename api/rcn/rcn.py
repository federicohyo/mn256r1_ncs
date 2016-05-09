############### author ##########
# federico corradi
# federico@ini.phys.ethz.ch
# Random Connected Network class mn256r1 
# ===============================

### ========================= import packages ===============================
import random
import numpy as np
import time
import pyNCS

class Rcn:
    def __init__(self, population,  cee=0.5 ):
        ### ========================= define what is needed to program the chip ====
        # resources
        self.available_neus = range(256)
        self.matrix_learning_rec = np.zeros([256,256])
        self.matrix_learning_pot = np.zeros([256,256])
        self.matrix_programmable_rec = np.zeros([256,256])
        self.matrix_programmable_w = np.zeros([256,256])
        self.matrix_programmable_exc_inh = np.zeros([256,256])
        # end resources
        # network parameters
        self.cee = cee
        self.rcn = population
        #end parameters 
        self.popsne = np.array([])
        self.pops_number = 1 
        self.popsne = []
        self.mypop_e = []
        self.setup = population.setup
        # populate chip
        self._populate_chip()
        # make connections
        #self._connect_populations()
        #this is AER retina connections
        self.matrix_programmable_w[1::4]  = 2 
        self.matrix_programmable_w[::4] = 1 
        self.matrix_programmable_exc_inh[:] = 1

    def _delete_me(self):
        del self

    ### ========================= functions ===================================

    def _connect_populations(self):
        # rcn with learning synapses
        self._connect_populations_programmable(self.popsne,self.popsne,self.cee,[2])
        self._connect_populations_programmable_inh(self.popsne,self.popsne,self.cee,[2])
        return 

    def _populate_chip(self):
        self.popsne = self._populate_neurons(len(self.rcn.soma.addr)) 
        # chip population
        self.mypop_e = pyNCS.Population('neurons', 'for fun') 
        self.mypop_e.populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.popsne))

    def _populate_neurons(self, n_neu ):
        #how many neurons we need to place?
        if(n_neu > len(self.available_neus)):
            print 'game over.. not so many neurons'
            raise Exception
        else:
            neu_ret = []
            neu_addr = self.rcn.soma.addr['neu']
            tf, index = self._ismember( neu_addr, np.array(self.available_neus))
            if(np.sum(tf) == len(neu_addr)):
                for i in range(len(neu_addr)):
                    neu_ret.append(self.available_neus.pop(neu_addr[i]-i))
        return neu_ret

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
        for pre in pop_pre:
            for post in pop_post:
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
        for pre in pop_pre:
            for post in pop_post:
                coin = np.random.rand()
                if(coin < connectivity):
                    #we connect this pre with this post
                    self.matrix_learning_rec[post,pre] = 1 
                    coin = np.random.rand()
                if(coin < pot):  
                    self.matrix_learning_pot[post,pre] = 1

    def _connect_populations_programmable(self, pop_pre,pop_post,connectivity,w):
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
        for pre in pop_pre:
            for post in pop_post:
                coin = np.random.rand()
                if(coin < connectivity):
                    #we connect this pre with this post
                    self.matrix_programmable_exc_inh[post,pre] = 1
                    self.matrix_programmable_rec[post,pre] = 1   
                    if(random_w):
                        self.matrix_programmable_w[post,pre] = np.random.randint(w_max)+w_min
                    else:
                        self.matrix_programmable_w[post,pre] = w[0]


    def _load_configuration(self, directory):
        '''
            load configuration from folder 
        '''
        self.popsne = np.loadtxt(directory+'popse.txt')
        self.popsni = np.loadtxt(directory+'popsi.txt')
        self.available_neus = np.loadtxt(directory+'conf_available_neus.txt')
        self.matrix_learning_rec = np.loadtxt(directory+'conf_matrix_learning_rec.txt')
        self.matrix_learning_pot = np.loadtxt(directory+'conf_matrix_learning_pot.txt')
        self.matrix_programmable_rec = np.loadtxt(directory+'conf_matrix_programmable_rec.txt')
        self.matrix_programmable_w = np.loadtxt(directory+'conf_matrix_programmable_w.txt')
        self.matrix_programmable_exc_inh = np.loadtxt(directory+'conf_matrix_matrix_programmable_exc_inh.txt')

    ####################################
    # upload configuration on mn256r1
    ###################################
    def upload_config(self):
        '''
        upload configuration matrices on the neuromorphic chip mn256r1
        '''
        self.setup.mapper._program_onchip_weight_matrix_programmable(self.matrix_programmable_w)
        self.setup.mapper._program_onchip_programmable_connections(self.matrix_programmable_rec)
        self.setup.mapper._program_onchip_exc_inh(self.matrix_programmable_exc_inh)
        self.setup.mapper._program_onchip_learning_state(self.matrix_learning_pot)
        self.setup.mapper._program_onchip_plastic_connections(self.matrix_learning_rec)
      
    #####################################
    # define stimulus functions
    #####################################
    def _create_stim_pop_e_virtual(self, popsne,mypop_e,id_pop,freq,duration, nsteps=5, t_start=5):
        timebins = np.linspace(t_start,duration+t_start,nsteps)
        n_neu = len(popsne[id_pop])
        stim_fixed = np.r_[[np.linspace(freq,freq,nsteps)]*len(popsne[id_pop])*1]
        stim_matrix = stim_fixed
        tf,index = self._ismember(mypop_e[id_pop].soma.laddr,popsne[id_pop])
        virtual_syn = mypop_e[id_pop][tf].synapses['virtual_exc'][3::4]
        spiketrain_pop_a = virtual_syn.spiketrains_inh_poisson(stim_matrix,timebins)
        return spiketrain_pop_a

    ########################################
    # you can now play with the attractors
    ########################################
    def kill_all(self, freq_stim = 500, duration = 1000, time_t = 1):
        '''
        send inhibitory kick to all inhibitory populations, in sequence with time_t delay between pops.
        '''
        for i in range(len(self.popsne)):
            spiketrain_pop_inh = self._create_stim_pop_e_virtual(self.popsni,self.mypop_i,i,freq_stim,duration, nsteps=5)
            self.setup.stimulate(spiketrain_pop_inh, send_reset_event=False, tDuration=duration+3000)
            time.sleep(time_t)

    def stimulate_seq(self, freq_stim = 500, duration = 1000, time_t = 1):
        '''
        send excitatory stimulation to all neuron pops, in sequence with time_t delay beween pops
        '''
        #create stimulus for excitatory populations
        for i in range(len(self.popsne)):
            spiketrain_pop_a = self._create_stim_pop_e_virtual(self.popsne,self.mypop_e,i,freq_stim,duration, nsteps=5)
            self.setup.stimulate(spiketrain_pop_a, send_reset_event=False, tDuration=duration+3000)
            time.sleep(time_t)

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
                index_neu = np.where(np.logical_and(spike_train[:,1] == n_neurons[i], np.logical_and(spike_train[:,0] >     bins[b] , spike_train[:,0] < bins[b+1] )) )
                mean_rate[i,b] = len(index_neu[0])*1000.0/(bins[b+1]-bins[b]) # time unit: ms
        return mean_rate

    def reconstruct_stim(self, rcnmon, bb=3, pixels_per_macro =11):
        stim = []
        rcn_raw_data = rcnmon.sl.raw_data()
        data = self.mean_neu_firing(rcn_raw_data,self.mypop_e.soma.addr['neu'], nbins=bb)
        for i in range(bb):
            img = np.reshape(data[0:pixels_per_macro**2,i],(pixels_per_macro,pixels_per_macro))
            stim.append(img)
        return stim

