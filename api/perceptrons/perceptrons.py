############### author ##########
# federico corradi
# fabio stefanini
# federico@ini.phys.ethz.ch
# Perceptrons Network class with features classification mn256r1 
# ===============================

### ========================= import packages ===============================
import random
import numpy as np
import time
import pyNCS
import matplotlib
from pylab import *

class Perceptrons:
    def __init__(self, perceptrons_pop, features_pop,  cee=1.0, n_class = 2):
        ### ========================= define what is needed to program the chip ====
        # resources
        self.available_neus = range(256)
        self.matrix_learning_rec = np.zeros([256,256])
        self.matrix_learning_pot = np.zeros([256,256])
        self.matrix_programmable_rec = np.zeros([256,256])
        self.matrix_programmable_w = np.zeros([256,256])
        self.matrix_programmable_exc_inh = np.zeros([256,256])
        self.matrix_broadcast = np.zeros([256,256])
        # end resources
        # network parameters
        self.cee = cee
        self.perceptrons_pop = perceptrons_pop
        self.features_pop = features_pop
        #end parameters 
        self.setup = perceptrons_pop.setup
        self.n_class = 2 # to be used for different teacher signals to know which perceptrons neurons corresponds to which class
        # populate chip
        self._connect_populations()
        # make connections
        #self._connect_populations()
        #this is AER retina connections
        self.matrix_programmable_w  = np.random.randint(1,size=(256,256))
        self.matrix_programmable_w[128:256,0:128] = 1
        self.matrix_programmable_exc_inh[0:128,:] = np.random.choice([0,1],size=(128,256))     
        self.matrix_programmable_exc_inh[128:256,0:128] = 1
        self.matrix_programmable_exc_inh[128:256,128:256] = 0

    def _delete_me(self):
        del self

    ### ========================= functions ===================================

    def _connect_populations(self):
        # rcn with learning synapses
        self._connect_populations_learning(self.features_pop.soma.addr,self.perceptrons_pop.soma.addr,self.cee,1)
        return 

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
    def upload_config(self, matrix_learning_state = True, matrix_exc_inh = True, matrix_programmable_w = True, matrix_programmable_rec = True, matrix_learning_rec = True ):
        '''
        upload configuration matrices on the neuromorphic chip mn256r1
        '''
        self.setup.chips['mn256r1'].load_parameters('biases/biases_program.biases')
        if matrix_programmable_w:
            self.setup.mapper._program_onchip_weight_matrix_programmable(self.matrix_programmable_w)
        if matrix_programmable_rec :
            self.setup.mapper._program_onchip_programmable_connections(self.matrix_programmable_rec)
        if matrix_exc_inh:
            self.setup.mapper._program_onchip_exc_inh(self.matrix_programmable_exc_inh)
        if matrix_learning_state:
            self.setup.mapper._program_onchip_learning_state(self.matrix_learning_pot)
        if matrix_learning_rec:
            self.setup.mapper._program_onchip_plastic_connections(self.matrix_learning_rec)

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

