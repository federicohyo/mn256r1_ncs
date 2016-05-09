# author federico corradi
# federico@ini.phys.ethz.ch
# retina class for tmpdiff as output of conv net from cx_cortex
# july 2014
import numpy as np
import math
import time
from pylab import *
import pyNCS

class Retina_cx():
    def __init__(self, inputpop):
        
        self.setup =  inputpop.setup 
        self.mapper = self.setup.mapper
        #### TODO: these parameters need to be loaded from the XML file setupfile
        self.mem_offset_mn256r1 = 400
        self.mem_offset_usb = 500
        self.mem_offset_retina = 600
        self.in_bit = 22 #defined in the xml setupfile
        self.possible_interfaces = 8 
        self.usb_interface = 1
        self.mn256r1_interface = 0
        self.retina_interface = 3
        self.matrix_w = np.zeros([256,256])
        #### end parameters
        self.fd = []
        self.pop_dest = inputpop 
        self.retina_map = np.zeros([128,128])
        self.dest_map = np.zeros([len(np.unique(inputpop.synapses['learning'].addr['neu']))+len(np.unique(inputpop.synapses['programmable'].addr['neu'])),len(inputpop.soma.addr)]) 
        #dest_map_learning = np.zeros([len(np.unique(inputpop.synapses['learning'].addr['neu'])),len(np.unique(inputpop.soma.addr['neu']))])
        #dest_map_programmable = np.zeros([len(np.unique(inputpop.synapses['programmable'].addr['neu'])),len(np.unique(inputpop.soma.addr['neu']))])
        self.dest_map_learning_flat = inputpop.synapses['learning'].paddr
        self.dest_map_programmable_flat = inputpop.synapses['programmable'].paddr
        self.dest_map[np.min(np.unique(inputpop.synapses['programmable'].addr['syntype']))-1:np.max(np.unique(inputpop.synapses['programmable'].addr['syntype'])),:] = np.reshape(self.dest_map_programmable_flat,[len(np.unique(inputpop.synapses['programmable'].addr))/len(np.unique(inputpop.synapses['programmable'].addr['neu'])), len(np.unique(inputpop.soma.addr['neu']))]) 
        self.dest_map[np.min(np.unique(inputpop.synapses['learning'].addr['syntype'])):np.max(np.       unique(inputpop.synapses['learning'].addr['syntype']))+1,:] = np.reshape(self.dest_map_learning_flat,[len(np.unique(inputpop.synapses['learning'].addr))/len(np.unique(inputpop.synapses['learning'].addr['neu'])), len(np.unique(inputpop.soma.addr['neu']))])
         
    def map_retina_to_mn256r1(self):
        '''
        this function set the mapper for the retina
        '''
        pre_address = np.linspace(0,2**14-1,2**14) 
        post_address = self.pop_dest.synapses['programmable'].paddr[::4]
        # 1 << 3 do not broadcast , 0  interface shift,  
        last4bits = 1<<3+0
        #do the mapping and for speed up we only open the device once
        self.mapper._Mappings__open_device()
        for i in range(len(pre_address)):
            self.mapper._program_memory(int(self.mem_offset_retina+8+pre_address[i]),int(post_address[i]),int(last4bits),open_device=False)        
        self.mapper._Mappings__close_device()
        self.mapper._program_detail_mapping(2**6)

    def map_retina_to_mn256r1_randomproj(self):
        ''' 
            highly hard coded 
        '''
        ###############################
        ### TEACHER SIGNALS FROM RETINA
        #teacher signals
        n_teach = 16 #number of teacher signals 
        #NB:  128/n_teach should result int
        
        retina_pix = np.zeros([128,128])
        for i in range(n_teach):
            retina_pix[0:10,i*((128/n_teach)):(i+1)*(128/n_teach)] = i+1
        teacher_reserved = np.where(retina_pix.flatten() != 0)[0]
        max_pix_teacher = np.max(teacher_reserved)
        
        #perceptrons
        perceptron_neurons = np.linspace(128,255,128)
        perceptron_per_teach = len(perceptron_neurons)/n_teach
        
        #map teacher signals
        self.mapper._Mappings__open_device()
        last4bits = 1<<3+0
        pre_teach = []
        post_teach = []
        this_perc_counter = 0
        for this_teach in range(n_teach):
            pre_address_teach = np.where(retina_pix.flatten() == this_teach+1)[0]
            #post neurons and synapses
            post_address_this_teach = []
            for this_perc in range(perceptron_per_teach):
                neu_index = (self.pop_dest.synapses['programmable'].addr['neu'] == perceptron_neurons[this_perc_counter])
                this_perc_counter = this_perc_counter+1
                syn_index = (self.pop_dest.synapses['programmable'].addr['syntype'] <= len(pre_address_teach)+256)
                tot_index = neu_index & syn_index
                post_address_teach = self.pop_dest.synapses['programmable'].paddr[tot_index]
                post_address_this_teach.append(post_address_teach)
            n_perceptrons_this_teach,b = np.shape(post_address_this_teach)
            pix_per_teach = len(pre_address_teach)/n_perceptrons_this_teach
            counter_pre = 0
            for this_perc in range(n_perceptrons_this_teach):
                for this_pix in range(pix_per_teach):
                    self.mapper._program_memory(int(self.mem_offset_retina+8+pre_address_teach[counter_pre]),int(post_address_this_teach[this_perc][this_pix]),int(last4bits),open_device=False)
                    pre_teach.append(pre_address_teach[counter_pre])
                    post_teach.append(post_address_this_teach[this_perc][this_pix])
                    counter_pre = counter_pre + 1
        self.mapper._Mappings__close_device()
        
        
        ##############
        ### FILTERS
        wij = np.r_[[np.random.permutation(np.concatenate([[1]*32, [-1]*32,  [0]*(122*122-64)])) for i in range(122)]]
        #
        npost_syn , npix_ret = np.shape(wij)
        matrix_exc_inh = np.zeros([256,256])
        pre_address = []
        post_address = []
        for this_syn_post in range(npost_syn):
            this_pre = np.where(wij[this_syn_post,:] != 0)[0]
            matrix_exc_inh[0:len(this_pre),this_syn_post] = wij[this_syn_post,:][this_pre]
            #this_post =  np.where(wij[:,0] != 0)[0]
            pre_address.append(this_pre)
            
        #build post addresses
        x,y = np.where(matrix_exc_inh == -1)
        matrix_exc_inh[x,y] = 0
    
        #post neurons and synapses
        neu_index = (self.pop_dest.synapses['programmable'].addr['neu'] <= 127)
        syn_index = (self.pop_dest.synapses['programmable'].addr['syntype'] <= 63+256)
        tot_index = neu_index & syn_index
        post_address = self.pop_dest.synapses['programmable'].paddr[tot_index]

        #do the mapping
        self.mapper._Mappings__open_device()
        last4bits = 1<<3+0
        pre = []
        post = []
        counter_post = 0
        for i in range(len(pre_address)):
            for j in range(len(pre_address[i])):
                self.mapper._program_memory(int(self.mem_offset_retina+8+(max_pix_teacher+1)+pre_address[i][j]),int(post_address[counter_post]),int(last4bits),open_device=False)
                pre.append(pre_address[i])
                post.append(post_address[i])
                counter_post = counter_post + 1
        self.mapper._Mappings__close_device()
        
        
        self.mapper._program_detail_mapping(2**6)
        self.retina_map = wij 
        matrix_exc_inh = matrix_exc_inh.T
        #add teacher signal on the matrix_exc inh
        matrix_exc_inh[int(np.min(perceptron_neurons)):int(np.max(perceptron_neurons)),0:len(pre_address_teach)] = 1
        
        #self.mapper._program_onchip_exc_inh(matrix_exc_inh)
        self.matrix_w[0:127,:] = 1 # = np.random.randint(0,4,size=(256,256))
        self.matrix_w[128:256,:] = 2
        #self.mapper._program_onchip_weight_matrix_programmable(self.matrix_w)

        return pre_teach, post_teach, pre, post


    def map_retina_sync(self, destpop,  ncol_retina = 5, neu_sync = 255):
    
        ###############################
        ### SYNC SIGNALS FROM RETINA
        num_col = ncol_retina
        pre_address = np.linspace(0,127*num_col,127*num_col+1)
        np.random.shuffle(pre_address)
 
        #post
        index_neu_sync = destpop.synapses['virtual_exc'].addr['neu'] == neu_sync
        syn_sync = destpop.synapses['virtual_exc'][index_neu_sync].paddr

        #map teacher signals
        self.mapper._Mappings__open_device()
        last4bits = 1<<3+0
        for this_perc in range(len(pre_address)):
            random_syn_post = int(np.random.choice(np.linspace(0,len(syn_sync)-1,len(syn_sync))))
            self.mapper._program_memory(int(self.mem_offset_retina+8+pre_address[this_perc]),syn_sync[random_syn_post],int(last4bits),open_device=False)
        self.mapper._Mappings__close_device()
        
        
    def map_retina_random_connectivity(self, destpop,  c = 0.2, syntype= 'programmable', ncol_sync = 5):
        '''
        this function set the mapper for the retina with random connectivity
        '''
        
        retina_pixels = 2**14
        num_square = len(self.pop_dest)
        pixels_per_square = np.floor(retina_pixels/num_square)
        index_list = []
        retina = np.zeros([128-ncol_sync,128-ncol_sync])
        for i in range(int(np.sqrt(num_square))):
            for j in range(int(np.sqrt(num_square))):
                this_square = 0.1*i+0.33*j+0.0537
                index_list.append(this_square)
                retina[j*int(np.sqrt(pixels_per_square)):(j+1)*int(np.sqrt(pixels_per_square)),i*int(np.sqrt(pixels_per_square)):(i+1)*int(np.sqrt(pixels_per_square))] = this_square
 
        #calculate random projections               
        n_neu_post = len(destpop.soma.addr)
        n_neu_pre = len(retina)
        n_connection = n_neu_pre*n_neu_post*c
        n_connection_per_macro = n_connection / num_square
        int_nconn_per_macro = int(np.floor(n_connection_per_macro))
        shift_sync = ncol_sync*128
        
        #now map macropixels
        self.mapper._Mappings__open_device()
        # 1 << 3 do not broadcast , 0  interface shift,  
        last4bits = 1<<3+0
        pre = []
        post = []
        if(num_square == 256):
            post_address = self.pop_dest.synapses[syntype].paddr[::2]
        if(num_square == 128):
            post_address = self.pop_dest.synapses[syntype].paddr[::2]  
        counter_post = 0
        if(len(post_address) > int_nconn_per_macro*num_square):
            print 'Connecting retina to reservoir with random projections...'
        else:
            print "repeat dest address..."
            mul_fact = float(int_nconn_per_macro*num_square)/len(post_address)
            mul_fact = mul_fact+1
            post_address = np.repeat(post_address, mul_fact)
            #print 'not possible to connect retina to reservoir, not enough post addresses'
            #raise Exception
        np.random.shuffle(post_address) #random projections    
        for i in range(int(num_square)):
            pre_address = np.where(retina.flatten() == index_list[i])[0]+shift_sync
            post_neuron = i
            if(len(pre_address) >= int_nconn_per_macro):
                loopover = int_nconn_per_macro
            else:
                loopover = len(pre_address)
            for i in range(loopover):
                self.mapper._program_memory(int(self.mem_offset_retina+8+pre_address[i]),int(post_address[counter_post]),int(last4bits),open_device=False)
                pre.append(pre_address[i])
                post.append(post_address[i])
                counter_post = counter_post + 1
        self.mapper._Mappings__close_device()
        self.mapper._program_detail_mapping(2**6)
        self.retina_map = retina
        
        return retina, pre, post
    
    def map_retina_to_mn256r1_macro_pixels(self, syntype = 'programmable'):
        '''
        this function set the mapper for the retina
        '''
        retina_pixels = 2**14
        num_square = len(self.pop_dest)
        pixels_per_square = np.floor(retina_pixels/num_square)
        index_list = []
        retina = np.zeros([128,128])
        for i in range(int(np.sqrt(num_square))):
            for j in range(int(np.sqrt(num_square))):
                this_square = 0.1*i+0.33*j+0.0537
                index_list.append(this_square)
                retina[j*int(np.sqrt(pixels_per_square)):(j+1)*int(np.sqrt(pixels_per_square)),i*int(np.sqrt(pixels_per_square)):(i+1)*int(np.sqrt(pixels_per_square))] = this_square
        #now map macropixels
        self.mapper._Mappings__open_device()
        # 1 << 3 do not broadcast , 0  interface shift,  
        last4bits = 1<<3+0
        pre = []
        post = []
        if(num_square == 256):
            post_address = self.pop_dest.synapses[syntype].paddr[::4]
        if(num_square == 128):
            post_address = self.pop_dest.synapses[syntype].paddr[::2]  
        counter_post = 0
        for i in range(int(num_square)):
            pre_address = np.where(retina.flatten() == index_list[i])[0]
            post_neuron = i
            for i in range(len(pre_address)):
                self.mapper._program_memory(int(self.mem_offset_retina+8+pre_address[i]),int(post_address[counter_post]),int(last4bits),open_device=False)
                pre.append(pre_address[i])
                post.append(post_address[i])
                counter_post = counter_post + 1
        self.mapper._Mappings__close_device()
        self.mapper._program_detail_mapping(2**6)
        self.retina_map = retina
        return retina, pre, post

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
        self.mapper._program_bulk_spec(self.retina_interface,2**self.usb_interface)
        #now we subtract the bit from the usb events defined in the setupfile
        self.mapper._program_offset(self.usb_interface,self.mem_offset_usb)
        self.mapper._program_memory(self.mem_offset_usb+0, 0, 0, prob=0)
        self.mapper._program_offset(self.mn256r1_interface, self.mem_offset_mn256r1)
        self.mapper._program_memory(self.mem_offset_mn256r1+0, 0, 0, prob=0)
        self.mapper._program_offset(self.retina_interface, self.mem_offset_retina)
        self.mapper._program_memory(self.mem_offset_retina+1,2**15,1,prob=0) #256
        #program address ranges
        self.mapper._program_address_range(self.usb_interface,(2**32)-1)
        self.mapper._program_address_range(self.mn256r1_interface,(2**32)-1)
        self.mapper._program_address_range(self.retina_interface,(2**32)-1)
        print "##### mapper with mn256r1 and retina initialization completed"

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

    def reconstruct_stim(self, rcnmon, bb=3, dim=16):
        stim = []
        rcn_raw_data = rcnmon.sl.raw_data()
        data = self.mean_neu_firing(rcn_raw_data,self.pop_dest.soma.addr['neu'], nbins=bb)
        for i in range(bb):
            img = np.reshape(data[0:len(self.pop_dest.soma.addr['neu']),0],(dim,dim))
            stim.append(img)
        return stim
        
    def recontruct_state(self):
        a = np.sum(self.state,axis=1)
        imshow(np.fliplr(a.reshape([16,16])),interpolation='nearest', origin='upper')
        colorbar()
        return
        
    def learn_stim_ret(duration, ifdc = 2.05e-06):
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_ret.biases')
        self.setup.chips['mn256r1'].set_parameter("IF_DC_P", ifdc)
        time.sleep(0.2)
        nsetup.mapper._program_detail_mapping(2**6)
        time.sleep(duration)
        nsetup.mapper._program_detail_mapping(2**7)
        return
        
        

