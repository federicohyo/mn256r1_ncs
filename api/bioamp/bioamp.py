# author federico corradi
# federico@ini.phys.ethz.ch
# bioamp class for bioamp + mn256r1 
# november 2014
import numpy as np
import math
import time
from pylab import *
import pyNCS

class Bioamp():
    def __init__(self, inputpop):
        
        self.chippop = inputpop
        self.setup =  inputpop.setup 
        self.mapper = self.setup.mapper
        #### TODO: these parameters need to be loaded from the XML file setupfile
        self.mem_offset_mn256r1 = 400
        self.mem_offset_usb = 500
        self.bioamp_up_address = 305
        self.bioamp_dn_address = 300
        self.bioamp_pulse_address = 310
        self.mem_offset_bioamp_up = 600
        self.mem_offset_bioamp_dn = 700 
        self.mem_offset_bioamp_pulse = 800
        self.in_bit = 22 #defined in the xml setupfile
        self.possible_interfaces = 8 
        self.usb_interface = 1
        self.mn256r1_interface = 0
        self.bioamp_up_interface = 3
        self.bioamp_dn_interface = 4
        self.bioamp_pulse_interface = 5
        self.matrix_w = np.zeros([256,256])
        self.pop_broadcast = pyNCS.Population("","")
        self.pop_broadcast.populate_all(self.setup, 'mn256r1', 'excitatory')
        
        self.matrix_learning_rec = np.zeros([256,256])
        self.matrix_learning_pot = np.zeros([256,256])
        self.matrix_programmable_rec = np.zeros([256,256])
        self.matrix_programmable_w = np.zeros([256,256])
        self.matrix_programmable_exc_inh = np.zeros([256,256])
        #### end parameters


    def _program_multicast(self, pre, post, memoffset, memoffset_interface, last4bits):
        ''' single pre to multi post with memory offset
            remember to open the device from outside
        '''

        if(np.size(pre) > 1):
            print 'this function map only one pre to multiple post neurons'
            return 
        if(len(post) < 2):
            print 'this function map only one pre to multiple post neurons'
            return 

        memory_loc_one_to_many = memoffset
        fan_out = len(post) #one irregular neuron
        
        #print 'programming multicast mapping'
        self.mapper._program_memory(int(memory_loc_one_to_many),int(fan_out),int(last4bits),open_device=True) 

        counter = 1
        for this_post in range(0,fan_out):
            self.mapper._program_memory(int(memory_loc_one_to_many+counter),int(post[this_post]),last4bits,open_device=True)
            #print 'at mem location ', int(memory_loc_one_to_many+counter), ' dest ', int(post[this_post])
            counter = counter+1

        self.mapper._program_memory(int(memoffset_interface+8+pre),int(memory_loc_one_to_many),0);
        #print 'now map input neuron ', int(memoffset_interface+8+pre)-int(memoffset_interface+8), ' to memory location ', int(memory_loc_one_to_many)

        memorypointer = memory_loc_one_to_many+counter

        return memorypointer

    def map_bioamp_features_neu(self, features_neu, syntype='programmable', n_syn_per_neu = 100, weight = 2):
        '''
        this function set the mapper for the bioamp for the perceptrons experiment
        '''
        pre_address_up = np.array([0]) #zero address
        pre_address_dn = np.array([0]) #the address is zero
        
        n_syn = n_syn_per_neu
        n_neu = len(features_neu)
        wei = np.zeros([n_syn,n_neu])
        syn_per_step = np.ceil(float(n_syn)/n_neu)
        counter = 0
        for this_neu in range(n_neu):
            wei[0:counter,this_neu] =  1
            counter = counter + syn_per_step 
               
        #receptive field is imshow(wei)
        matrix_w =  np.reshape(np.repeat(weight, n_syn*n_neu),([n_syn,n_neu]))        
        
        #loop over dest neus and create list of post-synaptic address
        tot_up_post_addr = []
        tot_dn_post_addr = []
        for index_neu,this_neu_post in enumerate(features_neu):
        
            if(syntype == 'programmable'):
                # up spikes
                index_neu_post_up = self.chippop.synapses['programmable'].addr['neu'] == this_neu_post
                index_syn_post_up = self.chippop.synapses['programmable'].addr['syntype'] < 256+np.sum(wei[:,index_neu])
                tot_syn = index_neu_post_up & index_syn_post_up
                post_address_up = self.chippop.synapses['programmable'][tot_syn].paddr
                
                tot_up_post_addr = np.append(tot_up_post_addr, post_address_up)
                
                #dn spikes
                dn_syn = n_syn - np.sum(wei[:,index_neu])
                index_neu_post_dn = self.chippop.synapses['programmable'].addr['neu'] == this_neu_post
                index_syn_post_dn = self.chippop.synapses['programmable'].addr['syntype'] < 256+dn_syn
                tot_syn = index_neu_post_dn & index_syn_post_dn
                post_address_dn = self.chippop.synapses['programmable'][tot_syn].paddr
                   
                tot_dn_post_addr = np.append(tot_dn_post_addr, post_address_dn)  
            if(syntype == 'virtual'):    
                # up spikes
                index_neu_post_up = self.chippop.synapses['virtual_exc'].addr['neu'] == this_neu_post
                index_syn_post_up = self.chippop.synapses['virtual_exc'].addr['syntype'] < 512+4
                tot_syn = index_neu_post_up & index_syn_post_up
                post_address_up = self.chippop.synapses['programmable'][tot_syn].paddr
                post_address_up = np.repeat(post_address_up,np.sum(wei[:,index_neu]))
              
                tot_up_post_addr = np.append(tot_up_post_addr, post_address_up)
                
                # dn spikes
                index_neu_post_dn = self.chippop.synapses['virtual_exc'].addr['neu'] == this_neu_post
                index_syn_post_dn = self.chippop.synapses['virtual_exc'].addr['syntype'] < 516+4
                tot_syn = index_neu_post_dn & index_syn_post_dn
                post_address_dn = self.chippop.synapses['programmable'][tot_syn].paddr
                dn_syn = n_syn - np.sum(wei[:,index_neu])
                post_address_dn = np.repeat(post_address_dn,dn_syn)
              
                tot_dn_post_addr = np.append(tot_dn_post_addr, post_address_dn)
                
            
        matrix_e_i = np.zeros([256,256])
        w_programmable = np.zeros([256,256])
        for i in range(len(features_neu)):
            matrix_e_i[i,0:n_syn] = wei[:,i]
            w_programmable[i,0:n_syn] = 2
        self.chippop.setup.mapper._program_onchip_exc_inh(matrix_e_i)
        self.chippop.setup.mapper._program_onchip_weight_matrix_programmable(w_programmable)
    
        # 1 << 3 do not broadcast, 0  interface shift,  
        last4bits = 1<<3+0
        memoffset_up = self.mem_offset_bioamp_up+10
        memoffset_up = self._program_multicast(pre_address_up, tot_up_post_addr, memoffset_up, self.mem_offset_bioamp_up, last4bits)
        memoffset_dn = self.mem_offset_bioamp_dn+10
        memoffset_dn = self._program_multicast(pre_address_dn, tot_dn_post_addr, memoffset_dn, self.mem_offset_bioamp_dn, last4bits)
        #self.mapper._Mappings__close_device()  ############################ CLOSE
        self.mapper._program_detail_mapping(2**self.bioamp_up_interface+2**self.bioamp_dn_interface)

        return matrix_e_i, w_programmable, tot_up_post_addr, tot_dn_post_addr 


    def map_bioamp_reservoir_broadcast_learning(self, n_columns = 3):
    
        pre_address_up = np.array([0]) #zero address
        pre_address_dn = np.array([0]) #the address is zero
        
        #use the broadcast
        post_address_up_t_inh = []
        post_address_up_t = []
        a = range(256)
        broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][0::256]
        matrix_b = np.ones([256,256])
        matrix_b[0:6,:] = 0
        self.setup.mapper._program_onchip_broadcast_programmable(matrix_b)
        
        post_address_up = []
        for this_col in range(n_columns):
            this_coladdr = broadcast_syn.paddr[this_col]
            post_address_up.append(this_coladdr)
  
        post_address_dn = []
        for this_col in range(n_columns,n_columns+2):
            this_coladdr = broadcast_syn.paddr[this_col]
            post_address_dn.append(this_coladdr)
        
        # 1 << 3 do not broadcast, 0  interface shift,  
        last4bits = 1<<3+0
        #do the mapping and for speed up we only open the device once
        #self.mapper._Mappings__open_device() ############################ OPEN 
        
        memoffset_up = self.mem_offset_bioamp_up+10
        memoffset_up = self._program_multicast(pre_address_up, post_address_up, memoffset_up, self.mem_offset_bioamp_up, last4bits)
 
        #self.mapper._Mappings__open_device() ############################ OPEN 
        memoffset_dn = self.mem_offset_bioamp_dn+10
        memoffset_dn = self._program_multicast(pre_address_dn, post_address_dn, memoffset_dn, self.mem_offset_bioamp_dn, last4bits)
                                       
        #self.mapper._Mappings__close_device()  ############################ CLOSE
        self.mapper._program_detail_mapping(2**self.bioamp_up_interface+2**self.bioamp_dn_interface)

           


    def map_bioamp_reservoir_broadcast(self, n_columns = 3):
    
        pre_address_up = np.array([0]) #zero address
        pre_address_dn = np.array([0]) #the address is zero
        
        #use the broadcast
        post_address_up_t_inh = []
        post_address_up_t = []
        a = range(256)
        broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][1::256]
        matrix_b = np.ones([256,256])
        self.setup.mapper._program_onchip_broadcast_programmable(matrix_b)
        
        post_address_up = []
        for this_col in range(n_columns):
            this_coladdr = broadcast_syn.paddr[this_col]
            post_address_up.append(this_coladdr)
  
        post_address_dn = []
        for this_col in range(n_columns,n_columns+2):
            this_coladdr = broadcast_syn.paddr[this_col]
            post_address_dn.append(this_coladdr)
        
        # 1 << 3 do not broadcast, 0  interface shift,  
        last4bits = 1<<3+0
        #do the mapping and for speed up we only open the device once
        #self.mapper._Mappings__open_device() ############################ OPEN 
        
        memoffset_up = self.mem_offset_bioamp_up+10
        memoffset_up = self._program_multicast(pre_address_up, post_address_up, memoffset_up, self.mem_offset_bioamp_up, last4bits)
 
        #self.mapper._Mappings__open_device() ############################ OPEN 
        memoffset_dn = self.mem_offset_bioamp_dn+10
        memoffset_dn = self._program_multicast(pre_address_dn, post_address_dn, memoffset_dn, self.mem_offset_bioamp_dn, last4bits)
            
        #memoffset_mn256r1 = self.mem_offset_mn256r1+10
        #memoffset_mn256r1 = self._program_multicast(100, post_address_dn_t, memoffset_mn256r1, self.mem_offset_mn256r1, last4bits)    
                                      
        #self.mapper._Mappings__close_device()  ############################ CLOSE
        self.mapper._program_detail_mapping(2**self.bioamp_up_interface+2**self.bioamp_dn_interface)

           
        #a = range(256)
        #broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][1::256][0:n_columns] #non plastic 
        #broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][0::256][0:n_columns] #plastic 
        #stim = broadcast_syn.spiketrains_poisson(450, duration=500)
        #program broadcast syn
        #matrix_b = np.ones([256,256])
        #self.setup.chips['mn256r1'].load_parameters('biases/biases_setbroadcast.biases')
        #self.setup.mapper._program_onchip_broadcast_programmable(matrix_b)
        
        

    def map_bioamp_reservoir(self, neudest_up, neudest_dn):
        '''
        this function set the mapper for the retina
        '''
        pre_address_up = np.array([0]) #zero address
        pre_address_dn = np.array([0]) #the address is zero
        
        #neudest_up = np.linspace(0,99,100)
        #neudest_dn = np.linspace(100,199,100)

        post_address_up_t_inh = []
        post_address_up_t = []
        for this_post in range(len(neudest_up)):
            index_neu_zero_up = self.chippop.synapses['virtual_exc'].addr['neu'] == neudest_up[this_post]
            post_address_up = self.chippop.synapses['virtual_exc'][index_neu_zero_up].paddr
            #post_address_up_inh = self.chippop.synapses['virtual_inh'][index_neu_zero_up].paddr

            post_address_up_t.append(post_address_up)
            #post_address_up_t_inh.append(post_address_up_inh) 
            
        post_address_dn_t_inh = []
        post_address_dn_t = []
        for this_post in range(len(neudest_dn)):
            index_neu_zero_dn = self.chippop.synapses['virtual_exc'].addr['neu'] == neudest_dn[this_post]
            post_address_dn = self.chippop.synapses['virtual_exc'][index_neu_zero_dn].paddr
            #post_address_dn_inh = self.chippop.synapses['virtual_inh'][index_neu_zero_dn].paddr

            post_address_dn_t.append(post_address_dn)
            #post_address_dn_t_inh.append(post_address_dn_inh) 
            
            
        #post_address_up_t = concatenate((post_address_up_t,post_address_dn_t_inh),axis=1)
        #post_address_dn_t = concatenate((post_address_up_t_inh, post_address_dn_t),axis=1) 

        post_address_up_t = np.reshape(post_address_up_t,len(post_address_up_t)*len(post_address_up_t[0]))
        post_address_dn_t = np.reshape(post_address_dn_t,len(post_address_dn_t)*len(post_address_dn_t[0]))

        # 1 << 3 do not broadcast, 0  interface shift,  
        last4bits = 1<<3+0
        #do the mapping and for speed up we only open the device once
        #self.mapper._Mappings__open_device() ############################ OPEN 
        
        memoffset_up = self.mem_offset_bioamp_up+10
        memoffset_up = self._program_multicast(pre_address_up, post_address_up_t, memoffset_up, self.mem_offset_bioamp_up, last4bits)
 
        #self.mapper._Mappings__open_device() ############################ OPEN 
        memoffset_dn = self.mem_offset_bioamp_dn+10
        memoffset_dn = self._program_multicast(pre_address_dn, post_address_dn_t, memoffset_dn, self.mem_offset_bioamp_dn, last4bits)
            
        #memoffset_mn256r1 = self.mem_offset_mn256r1+10
        #memoffset_mn256r1 = self._program_multicast(100, post_address_dn_t, memoffset_mn256r1, self.mem_offset_mn256r1, last4bits)    
                                      
        #self.mapper._Mappings__close_device()  ############################ CLOSE
        self.mapper._program_detail_mapping(2**self.bioamp_up_interface+2**self.bioamp_dn_interface)


    def map_bioamp_onetomany(self, neudest_up, neudest_dn):
        '''
        this function set the mapper for the bioamp
        '''
        pre_address_up = np.array([0]) #zero address
        pre_address_dn = np.array([0]) #the address is zero
        
        index_neu_zero_up = self.chippop.synapses['virtual_exc'].addr['neu'] == neudest_up
        post_address_up = self.chippop.synapses['virtual_exc'][index_neu_zero_up].paddr
        post_address_up_inh = self.chippop.synapses['virtual_inh'][index_neu_zero_up].paddr

        index_neu_zero_dn = self.chippop.synapses['virtual_exc'].addr['neu'] == neudest_dn
        post_address_dn = self.chippop.synapses['virtual_exc'][index_neu_zero_dn].paddr
        post_address_dn_inh = self.chippop.synapses['virtual_inh'][index_neu_zero_dn].paddr

        post_address_up_t = concatenate((post_address_up,post_address_dn_inh),axis=1)
        post_address_dn_t = concatenate((post_address_up_inh, post_address_dn),axis=1) 

        # 1 << 3 do not broadcast, 0  interface shift,  
        last4bits = 1<<3+0
        #do the mapping and for speed up we only open the device once
        #self.mapper._Mappings__open_device() ############################ OPEN 
        
        memoffset_up = self.mem_offset_bioamp_up+10
        memoffset_up = self._program_multicast(pre_address_up, post_address_up_t, memoffset_up, self.mem_offset_bioamp_up, last4bits)
 
        #self.mapper._Mappings__open_device() ############################ OPEN 
        memoffset_dn = self.mem_offset_bioamp_dn+10
        memoffset_dn = self._program_multicast(pre_address_dn, post_address_dn_t, memoffset_dn, self.mem_offset_bioamp_dn, last4bits)
                                      
        #self.mapper._Mappings__close_device()  ############################ CLOSE
        self.mapper._program_detail_mapping(2**self.bioamp_up_interface+2**self.bioamp_dn_interface)

         
    def map_bioamp_delta_single_dest(self, neudest):
        '''
        this function set the ad delta to one single destination neuron
        '''
        pre_address_up = np.array([0]) #zero address
        pre_address_dn = np.array([0]) #the address is zero
        
        index_neu_zero_up = self.chippop.synapses['virtual_exc'].addr['neu'] == neudest
        post_address_up = self.chippop.synapses['virtual_exc'][index_neu_zero_up].paddr
        post_address_up_inh = self.chippop.synapses['virtual_inh'][index_neu_zero_up].paddr

        # 1 << 3 do not broadcast , 0  interface shift,  
        last4bits = 1<<3+0
        #do the mapping and for speed up we only open the device once
        self.mapper._Mappings__open_device() ############################ OPEN 
        
        for i in range(len(post_address_up)):
            self.mapper._program_memory(int(self.mem_offset_bioamp_up+8+pre_address_up[0]),int(post_address_up[i]),int(last4bits),open_device=False)       
        for i in range(len(post_address_up_inh)):
            self.mapper._program_memory(int(self.mem_offset_bioamp_dn+8+pre_address_dn[0]),int(post_address_up_inh[i]),int(last4bits),open_device=False)                   
                           
        self.mapper._Mappings__close_device()  ############################ CLOSE
        self.mapper._program_detail_mapping(2**self.bioamp_up_interface+2**self.bioamp_dn_interface)


    def _init_fpga_mapper(self):
        print "clear mapper registers"
        #clear registers
        for i in range(self.possible_interfaces):
            self.mapper._program_address_range(int(i),(2**32)-1)
            self.mapper._program_offset(int(i),int(0))
            self.mapper._program_bulk_spec(int(i),0)
        self.mapper._program_detail_mapping(0)
        #clear memory
        print "program bulk specifications"
        #default bulk specs chip->usb and usb->chip
        self.mapper._program_bulk_spec(self.usb_interface,2**self.mn256r1_interface)
        self.mapper._program_bulk_spec(self.mn256r1_interface,2**self.usb_interface)
        self.mapper._program_bulk_spec(self.bioamp_up_interface,2**self.usb_interface)
        self.mapper._program_bulk_spec(self.bioamp_dn_interface,2**self.usb_interface)
        self.mapper._program_bulk_spec(self.bioamp_pulse_interface,2**self.usb_interface)

        self.mapper._program_offset(self.usb_interface,self.mem_offset_usb)
        self.mapper._program_memory(self.mem_offset_usb+0, 0, 0, prob=0)
        self.mapper._program_offset(self.mn256r1_interface, self.mem_offset_mn256r1)
        self.mapper._program_memory(self.mem_offset_mn256r1+0, 0, 0, prob=0)
        self.mapper._program_offset(self.bioamp_up_interface, self.mem_offset_bioamp_up)
        self.mapper._program_memory(self.mem_offset_bioamp_up+1,self.bioamp_up_address,0,prob=0) #256
        self.mapper._program_offset(self.bioamp_dn_interface, self.mem_offset_bioamp_dn)
        self.mapper._program_memory(self.mem_offset_bioamp_dn+1,self.bioamp_dn_address,0,prob=0) #256
        self.mapper._program_offset(self.bioamp_pulse_interface, self.mem_offset_bioamp_pulse)
        self.mapper._program_memory(self.mem_offset_bioamp_pulse+1,self.bioamp_pulse_address,0,prob=0) #256
        
        #program address ranges
        self.mapper._program_address_range(self.usb_interface,(2**32)-1)
        self.mapper._program_address_range(self.mn256r1_interface,(2**32)-1)
        self.mapper._program_address_range(self.bioamp_up_interface,(2**32)-1)
        self.mapper._program_address_range(self.bioamp_dn_interface,(2**32)-1)
        self.mapper._program_address_range(self.bioamp_pulse_interface,(2**32)-1)
        
        print "##### mapper with mn256r1 and bioamp initialization completed"



    def _ismember(self,a,b):
        '''
        as matlab: ismember
        '''
        # tf = np.in1d(a,b) # for newer versions of numpy
        tf = np.array([i in b for i in a])
        u = np.unique(a[tf])
        index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
        return tf, index

    def _show_mapping(self, pre_address, post_address):
        dd = np.reshape(self.dest_map,[512*256])
        tf, index = self._ismember(dd,post_address)
        outmatrix = np.zeros([512*256])
        outmatrix[index] = pre_address
        self.dest_map[index] = pre_address
        outmatrix = np.reshape(outmatrix,np.shape(self.dest_map))
        imshow(outmatrix)
        return outmatrix

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


    def connect_perceptrons_with_features(self, perceptrons_pop, features_pop,  cee=1.0, n_class = 2):

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

    ### ========================= functions ===================================

    def _connect_populations(self):
        # rcn with learning synapses
        self._connect_populations_learning(self.features_pop.soma.addr,self.perceptrons_pop.soma.addr,self.cee,1)
        return 


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
      


