############### author ##########
# Central Pattern Generator
# elisa donati sssup
# ===============================

### ========================= import packages ===============================
import random
import numpy as np
import time
import pyNCS


class Cpg:
    def __init__(self, chippop,  cee=1, cei=1, n_segments=1, n_neu_osc = 4 ):
        # resources
        self.available_neus = range(256)
        self.matrix_learning_rec = np.zeros([256,256])
        self.matrix_learning_pot = np.zeros([256,256])
        self.matrix_programmable_rec = np.zeros([256,256])
        self.matrix_programmable_w = np.zeros([256,256])
        self.matrix_programmable_exc_inh = np.zeros([256,256])
        self.setup = chippop.setup
        self.mapper = self.setup.mapper
        #end resources
        # network parameters
        self.cee = cee
        self.cei = cei
        self.n_segments = n_segments
        self.n_neu_osc = n_neu_osc 
        self.chippop = chippop
        self.npop = 10
        self.nneuinpop = 5
        self.cpg = [pyNCS.Population() for i in range(self.npop)]
        for i in range(self.npop):
            self.cpg[i].populate_by_id(self.setup, 'mn256r1', 'excitatory', np.linspace(i*5,(((i+1)*5)-1), 5))
        #population = [pyNCS.Population() for i in range(npop)]
        #population[0] = population.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,))
        #
        #end parameters
         
     
        self.popsosc = np.array([])
        self.popsnosc1 = np.array([])
        self.pop_mn_d = np.array([])
        self.pop_mn_s = np.array([])
        self.pop_ein_s = np.array([])
        self.pop_ein_d = np.array([])
        self.pop_lin_d = np.array([])
        self.pop_lin_s = np.array([])
        self.pop_cn_s = np.array([])
        self.pop_cn_d = np.array([])
        self._populate_chip()
        self._connect_populations()
        self._connect_populations_oscillator()
        self._input_oscillator()
        #self._create_stim_virtual(self.popsosc, self.popsosc1)
 
    def _delete_me(self):
         del self  
  ### ========================= functions ===================================

    
    def _populate_chip(self):
    
        self.popsosc = (self.cpg[0].soma.addr['neu'])
        self.popsosc1 = (self.cpg[1].soma.addr['neu'])
        self.pop_mn_d = (self.cpg[2].soma.addr['neu'])
        self.pop_mn_s = (self.cpg[3].soma.addr['neu']) 
        self.pop_ein_s = (self.cpg[8].soma.addr['neu'])
        self.pop_ein_d = (self.cpg[9].soma.addr['neu'])
        self.pop_lin_d = (self.cpg[7].soma.addr['neu'])
        self.pop_lin_s = (self.cpg[6].soma.addr['neu'])
        self.pop_cn_s = (self.cpg[4].soma.addr['neu'])
        self.pop_cn_d = (self.cpg[5].soma.addr['neu'])
        # chip population
        #oscillator population
        pop_osc = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_osc[i].populate_by_id(self.setup,'mn256r1','excitatory', np.array(self.popsosc))for i in range(self.n_segments)]
        pop_osc1 = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_osc1[i].populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.popsosc1)) for i in range(self.n_segments)]
        #mn population dx
        pop_mn_dx = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_mn_dx[i].populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.pop_mn_d)) for i in range(self.n_segments)]
        #mn population sx
        pop_mn_sx = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_mn_sx[i].populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.pop_mn_s)) for i in range(self.n_segments)]
        #ein population sx
        pop_ein_sx = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_ein_sx[i].populate_by_id(self.setup,'mn256r1','excitatory', np.array(self.pop_ein_s)) for i in range(self.n_segments)]
        #ein population dx
        pop_ein_dx = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_ein_dx[i].populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.pop_ein_d)) for i in range(self.n_segments)]
        #lin population dx
        pop_lin_dx = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_lin_dx[i].populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.pop_lin_d)) for i in range(self.n_segments)]
        #lin population sx
        pop_lin_sx = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_lin_sx[i].populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.pop_lin_s)) for i in range(self.n_segments)]
        #cn population sx
        pop_cn_sx = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_cn_sx[i].populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.pop_cn_s)) for i in range(self.n_segments)]
        #cn population dx
        pop_cn_dx = [pyNCS.Population('neurons', 'oscillator input') for i in range(self.n_segments)]
        [pop_cn_dx[i].populate_by_id(self.setup,'mn256r1','excitatory',np.array(self.pop_cn_d)) for i in range(self.n_segments)]

    def _connect_populations_oscillator(self):
        for i in range( self.n_neu_osc-1):
            self.connect_populations_programmable([self.popsosc[i]],[self.popsosc[i+1]],1,[2])
            self.connect_populations_programmable([self.popsosc1[i]],[self.popsosc1[i+1]],1,[2])
            if ( i ==  self.n_neu_osc-2):
                self.connect_populations_programmable([self.popsosc[i]],[self.popsosc[0]],1,[2])
                self.connect_populations_programmable([self.popsosc1[i]],[self.popsosc1[0]],1,[2])
# ===== inhibitory connections between populations of oscillators == inh
        for i in range(self.n_neu_osc):
            self.connect_populations_programmable_inh([self.popsosc[i]],[self.popsosc1[i]],1,[2])
            self.connect_populations_programmable_inh([self.popsosc1[i]],[self.popsosc[i]],1,[2])
        for i in range(self.n_neu_osc-1):
            self.connect_populations_programmable_inh([self.popsosc[i]],[self.popsosc1[i+1]],1,[2])
            self.connect_populations_programmable_inh([self.popsosc1[i]],[self.popsosc[i+1]],1,[2])
        return    

    def connect_populations_programmable_inh(self, pop_pre, pop_post,connectivity,w):
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

    def connect_populations_learning(self, pop_pre,pop_post,connectivity,pot):
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

    def connect_populations_programmable(self, pop_pre,pop_post,connectivity,w):
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


    def _connect_populations(self):
        for i in range(self.n_segments):
           #EIN -> EIN
            self.connect_populations_programmable(self.pop_ein_d,self.pop_ein_d,self.cee,[2,2]) 
            self.connect_populations_programmable(self.pop_ein_s,self.pop_ein_s,self.cee,[2,2]) 
            ##LIN -> LIN
            self.connect_populations_programmable(self.pop_lin_d,self.pop_lin_d,self.cee,[2,2]) 
            self.connect_populations_programmable(self.pop_lin_s,self.pop_lin_s,self.cee,[2,2]) 
            ##CN -> CN
            self.connect_populations_programmable(self.pop_cn_d,self.pop_cn_d,self.cee,[2,2]) 
            self.connect_populations_programmable(self.pop_cn_s,self.pop_cn_s,self.cee,[2,2]) 
            ##MN -> MN
            self.connect_populations_programmable(self.pop_mn_d,self.pop_mn_d,self.cee,[2,2]) 
            self.connect_populations_programmable(self.pop_mn_s,self.pop_mn_s,self.cee,[2,2])    
        for i in range(self.n_segments):
            for i in range(self.nneuinpop-1):
                ##EIN -> MN
                self.connect_populations_programmable(self.pop_ein_d,self.pop_mn_d,1,[2,2]) 
                self.connect_populations_programmable(self.pop_ein_s,self.pop_mn_s,1,[2,2]) 
                ##EIN -> LIN
                self.connect_populations_programmable(self.pop_ein_d,self.pop_lin_d,1,[2,2]) 
                self.connect_populations_programmable(self.pop_ein_s,self.pop_lin_s,1,[2,2]) 
                ##EIN -> CN
                self.connect_populations_programmable(self.pop_ein_d,self.pop_cn_d,1,[2,2]) 
                self.connect_populations_programmable(self.pop_ein_s,self.pop_cn_s,1,[2,2]) 
                ##LIN -> CN
                #self.connect_populations_programmable(self.pop_lin_d,self.pop_cn_d,1,[2,2]) 
                #self.connect_populations_programmable(self.pop_lin_s,self.pop_cn_s,1,[2,2])
                ##EIN-> EIN
                ##self.connect_populations_programmable(self.pop_ein_d[i],self.pop_ein_d[i],1,[2,2]) 
                ##self.connect_populations_programmable(self.pop_ein_s[i],self.pop_ein_s[i],1,[2,2]) 

                ##CN -> inhibit all 
                ##CNdx -> inh sx
                self.connect_populations_programmable_inh(self.pop_cn_d,self.pop_mn_s,1,[2,2]) 
                self.connect_populations_programmable_inh(self.pop_cn_d,self.pop_lin_s,1,[2,2]) 
                self.connect_populations_programmable_inh(self.pop_cn_d,self.pop_cn_s,1,[2,2]) 
                self.connect_populations_programmable_inh(self.pop_cn_d,self.pop_ein_s,1,[2,2]) 
                ##CNsx -> inh dx
                self.connect_populations_programmable_inh(self.pop_cn_s,self.pop_mn_d,1,[2,2]) 
                self.connect_populations_programmable_inh(self.pop_cn_s,self.pop_lin_d,1,[2,2]) 
                self.connect_populations_programmable_inh(self.pop_cn_s,self.pop_cn_d,1,[2,2]) 
                self.connect_populations_programmable_inh(self.pop_cn_s,self.pop_ein_d,1,[2,2])     
        return

    def _input_oscillator(self):
        for i in range(self.n_neu_osc-1):
            self.connect_populations_programmable(self.popsosc,self.pop_ein_d,1,[1])
            self.connect_populations_programmable(self.popsosc1,self.pop_ein_s,1,[1])        
        

    def _populate_neurons(self, neu_addr ):
        #how many neurons we need to place?
        if(len(neu_addr) > len(self.available_neus)):
            print 'game over.. not so many neurons'
            raise Exception
        else:
            neu_cpg = []
            #neu_addr = self.cpg.soma.addr['neu']
            tf, index = self._ismember( neu_addr, np.array(self.available_neus))
            if(np.sum(tf) == len(neu_addr)):
                for i in range(len(neu_addr)):
                    neu_cpg.append(self.available_neus.pop(neu_addr[i]-i))
        return neu_cpg

      
    ####################################
    # upload configuration on mn256r1
    ###################################
    def upload_config(self):
        '''
            pload configuration matrices on the neuromorphic chip mn256r1
        '''
        print "Program recurrent plastic connections ..."
        self.setup.mapper._program_onchip_plastic_connections(self.matrix_learning_rec)
        print "Program recurrent plastic connections ..."      
        self.setup.mapper._program_onchip_programmable_connections(self.matrix_programmable_rec)
        print "Program connections weights ..."
        self.setup.mapper._program_onchip_weight_matrix_programmable(self.matrix_programmable_w)
        print "Program connections exc/inh ..."  
        self.setup.mapper._program_onchip_exc_inh(self.matrix_programmable_exc_inh)

    def _ismember(self, a,b):
        '''
        as matlab: ismember
        '''
        # tf = np.in1d(a,b) # for newer versions of numpy
        tf = np.array([i in b for i in a])
        u = np.unique(a[tf])
        index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
        return tf, index


    def _config_default_bias(self):
        #==================== load biases
        self.setup.chips['mn256r1'].load_parameters('biases/bias_cpg_set14')
        

    #def _create_stim_virtual(self, popsosc, popsosc1):
        ##create stimulus for excitatory motor neurons
        #dur = 100
        #freq_stim = 350  

        #syn_  = popsosc[0].synapses['virtual_exc'][3::4]
        #spiketrain = syn_.spiketrains_regular(freq_stim ,duration=dur)
        #self.setup.stimulate(spiketrain, send_reset_event=False, duration=dur)   

        #syn_  = popsosc1[1].synapses['virtual_exc'][3::4]
        #spiketrain = syn_.spiketrains_regular(freq_stim ,duration=dur)
        #self.setup.stimulate(spiketrain, send_reset_event=False, duration=dur) 
        #return spiketrain   




