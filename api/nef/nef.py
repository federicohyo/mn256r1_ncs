############### author ####################
# federico corradi
# federico@ini.phys.ethz.ch
# Neural Engineering Framework with mn256r1 
# =========================================

### ========================= import packages ===============================
import random
import numpy as np
import time
import pyNCS
import matplotlib
from pylab import *
import cvxpy as cx

class Nef:
    def __init__(self,pops,pop_broadcast,encoders):
        ### ========================= define what is needed to program the chip ====
        # resources
        self.available_neus = range(256)
        self.matrix_learning_rec = np.zeros([256,256])
        self.matrix_learning_pot = np.zeros([256,256])
        self.matrix_programmable_rec = np.zeros([256,256])
        self.matrix_programmable_w = np.zeros([256,256])
        self.matrix_programmable_exc_inh = np.zeros([256,256])
        self.matrix_programmable_broadcast = np.zeros([256,256])
        # end resources
        #end parameters 
        self.encoders = encoders
        self.pops = pops
        self.pop_broadcast = pop_broadcast
        self.setup = pops[0].setup
        self.chip = self.setup.chips['mn256r1']
        self.biases = [] # nef biases in this case is the ratio of exc / inh
        self.syn_eff = np.zeros([256,256,4])
        self.syn_eff_virt = []
        #mapper parameters
        self.mem_offset_mn256r1 = 300
        self.mem_offset_usb = 1000
        self.mem_offset_used = 0 
        self.in_bit = 22 #defined in the xml setupfile
        self.possible_interfaces = 8
        self.usb_interface = 1
        self.mn256r1_interface = 0
        self.bit_input_setupfile = 21 #setupfile.xml -1

    def _init_fpga_mapper(self):
        ####
        print "1) clear mapper registers"
        #clear registers
        for i in range(self.possible_interfaces):
            self.setup.mapper._program_address_range(int(i),(2**32)-1)
            self.setup.mapper._program_offset(int(i),int(0))
            self.setup.mapper._program_bulk_spec(int(i),0)
        self.setup.mapper._program_detail_mapping(0)
        #clear memory
        print "2) warning: not cleaning mapper memory, we assume it is empty"
        print "3) program bulk specifications"
        #default bulk specs chip->usb and usb->chip
        self.setup.mapper._program_bulk_spec(self.usb_interface,2**self.mn256r1_interface)
        self.setup.mapper._program_bulk_spec(self.mn256r1_interface,2**self.usb_interface)
        #now we subtract the bit from the usb events defined in the setupfile
        self.setup.mapper._program_offset(self.usb_interface,self.mem_offset_usb)
        self.setup.mapper._program_memory(self.mem_offset_usb+0, 0, 0, prob=0)
        self.setup.mapper._program_offset(self.mn256r1_interface, self.mem_offset_mn256r1)
        self.setup.mapper._program_memory(self.mem_offset_mn256r1+0, 0, 0, prob=0)
        #program address ranges
        self.setup.mapper._program_address_range(self.usb_interface,(2**32)-1)
        self.setup.mapper._program_address_range(self.mn256r1_interface,(2**32)-1)
        print "4) clear onchip recurrent connections"
        matrix = np.zeros([256,512])
        self.setup.mapper._program_onchip_recurrent(matrix)
        print "##### mapper initialization completed"

 
    def _delete_me(self):
        del self

    def upload_config(self):
        '''
        upload configuration matrices on the neuromorphic chip mn256r1
        '''
        self.setup.mapper._program_onchip_weight_matrix_programmable(self.matrix_programmable_w)
        self.setup.mapper._program_onchip_programmable_connections(self.matrix_programmable_rec)
        self.setup.mapper._program_onchip_exc_inh(self.matrix_programmable_exc_inh)
        self.setup.mapper._program_onchip_learning_state(self.matrix_learning_pot)
        self.setup.mapper._program_onchip_plastic_connections(self.matrix_learning_rec)
      
    def _ismember(self, a, b):
        '''
        as matlab: ismember
        '''
        # tf = np.in1d(a,b) # for newer versions of numpy
        tf = np.array([i in b for i in a])
        u = np.unique(a[tf])
        index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
        return tf, index
 
    def mean_neu_firing(self, spike_train, n_neurons,nbins=10):
        simulation_time = [np.min(spike_train[:,0]),np.max(spike_train[:,0])]
        un, bins = np.histogram(simulation_time,nbins)
        mean_rate = np.zeros([len(n_neurons),nbins])
        for b in range(nbins):
            #simulation_time = [np.min(spike_train[0][:]), np.max(spike_train[0][:])]
            for i in range(len(n_neurons)):
                index_neu = np.where(np.logical_and(spike_train[:,1] == n_neurons[i], np.logical_and(spike_train[:,0] >     bins[b] , spike_train[:,0] < bins[b+1] )) )
                mean_rate[i,b] = (len(index_neu[0])*1000.)/(bins[b+1]-bins[b]) # time unit: ms
        return mean_rate

    def generate_biases(self):
        #bias is the ration of exc / inh input synapses
        self.biases = [np.random.random(len(self.pops[i].soma.addr)) for i in range(len(self.pops))]

    def program_encoders(self):
        ''' set encoders by choosing different taus for neurons with encoders +1 or -1
        '''
        self.chip.configurator._set_all_neu_tau2()
        for this_g in range(len(self.encoders)):
            index_pop_tau1 = np.where(self.encoders[this_g] == -1)[0]
            self.chip.configurator._set_neuron_tau1(self.pops[this_g].soma[index_pop_tau1].addr['neu']) 

    def program_input_weights(self,nsyn = 256, w = 1, exc_col = 255, inh_col=254 ):
        ''' program programmable weight matrix with weight w 
        '''
        if(len(self.biases) == 0):
            print 'Please define biases, by running nef.generate_biases()'
            raise Exception
        else:
            for this_pop in range(len(self.pops)):
                neu_nsyn_exc = np.round(nsyn*self.biases[this_pop])
                neu_nsyn_inh = nsyn - neu_nsyn_exc  
                # we now make programmable exc_inh syn_matrix
                index_neu = self.pops[this_pop].soma.addr['neu']
                for this_neu in range(len(index_neu)):
                    if(self.encoders[this_pop][this_neu] == -1):
                        syn_line = np.repeat(0,nsyn) # all inhibitory
                    else:
                        syn_line = np.repeat(1,nsyn) # all exc
                        #exc_syn_this_neu = np.repeat(1,neu_nsyn_exc[this_neu]) 
                        #inh_syn_this_neu = np.repeat(0,neu_nsyn_inh[this_neu])
                        #lots exc and a bit inh
                        #syn_line = np.concatenate([exc_syn_this_neu,inh_syn_this_neu])
                    self.matrix_programmable_exc_inh[index_neu[this_neu],0:nsyn] = syn_line
        print "programming synaptic matrix exc / inh ..."
        self.exc_prog_syn = [exc_col+256] 
        self.inh_prog_syn = [inh_col+256]
        self.matrix_programmable_exc_inh[:,exc_col] = 1 
        self.matrix_programmable_exc_inh[:,inh_col] = 0 
        self.setup.mapper._program_onchip_exc_inh(self.matrix_programmable_exc_inh)
        print "done"
        print "programming weights"
        self.matrix_programmable_w = np.zeros([256,256])+w
        self.setup.mapper._program_onchip_weight_matrix_programmable(self.matrix_programmable_w)
        print "done"
        print "programming broadcast"
        self.matrix_programmable_broadcast[:,0:nsyn] = 1 
        self.setup.mapper._program_onchip_broadcast_programmable(self.matrix_programmable_broadcast)
        print "done"

    def measure_tuning_curves(self, min_freq=0, max_freq=200, nsteps = 10, nsyn=256, step_dur=500, b_cast_programmable = True, debug = False, do_plot = False):
        ''' measure tuning curves for all populations using broadcast stimulation '''
        x_values = np.linspace(min_freq,max_freq,nsteps)        
        timebins = np.linspace(0, step_dur*nsteps+1,nsteps+1)
        if(b_cast_programmable):
            # we only broadcast on programmable syn 0/256
            # address programmable broadcast should be 
            # address_broadcast = np.linspace(133120,133120+255,256) #learning synapses
            # address_broadcast = np.linspace(133376,133631,256) #programmable synapses
            # good luck in understanding this...
            a = range(256)
            if(nsyn == 128):
                broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][1::512][0] #programmable columns synapses 
            elif(nsyn == 256):
                broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][1::256][0]
            #broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][0::256] #learning columns synapses
        else:
            broadcast_syn = self.pops[a].synapses['broadcast'][::256] #only learning
        for this_phase in range(len(x_values)):
            this_stim = broadcast_syn.spiketrains_regular(x_values[this_phase],t_start=timebins[this_phase],duration=step_dur)
            if(this_phase == 0):
                stim = this_stim
            else:
                stim = pyNCS.pyST.merge_sequencers(stim,this_stim)

        out = self.setup.stimulate(stim,send_reset_event=False,duration=max(timebins))
        out = out[self.pops[0].soma.channel]
        raw_data = out.raw_data()
        offset_t = np.min(out.raw_data()[:,0])
        raw_data[:,0] = raw_data[:,0]-offset_t
        stop_time = timebins[-2]
        index_to_remove = np.where(raw_data[:,0] > stop_time)[0]
        raw_data = np.delete(raw_data,index_to_remove,axis=0)
        timebins = np.delete(timebins,nsteps)
        freq = self.mean_neu_firing(raw_data,range(256),nbins=nsteps) 
        self.tuning_curves = freq
        self.x_values = x_values
        self.tuning_bins = timebins
        if(do_plot == True):
            figure()
            for p in freq:
                plot(x_values,p,'o-')

        if(debug == True):
            return stim,out,timebins,nsteps
        else:
            return

    def plot_tuning_curves(self):
        '''
        guess!
        '''
        try:
            self.tuning_curves
        except:
            print "you must first measure the tuning curve with nef.measure_tuning_curves()!!!"
            return
        else:
            for this_pop in range(len(self.pops)):
                figure()
                hold(True)
                for this_enc in range(len(self.encoders[this_pop])):
                    this_neu = self.pops[this_pop][this_enc].soma.addr['neu'] 
                    if(self.encoders[this_pop][this_enc] < 0 ):
                        plot(self.x_values,self.tuning_curves[this_neu,:][0],'bo-')
                    else:
                        plot(self.x_values,self.tuning_curves[this_neu,:][0],'ro-')
                xlabel(r'$\nu_{in} [Hz]$',fontsize=18)
                ylabel(r'$\nu_{out} [Hz]$',fontsize=18)
                string_title = ('Tuning curves for population ' + str(this_pop))
                title(string_title)
            return 

    def function_exp(self,x,expo):
        self.exp = expo
        return x**expo

    def find_decoders(self,values):
        '''
        find optimal linear decoders
        '''
        A=np.array(self.tuning_curves)
        gamma=np.dot(A, A.T) #I diagonal noise add I = eye(len(gamma))
        upsilon=np.dot(A, values)
        ginv=np.linalg.pinv(gamma)
        decoders=np.dot(ginv,upsilon)
        return decoders

    def encode_function(self,find_decoders=True,nsteps=10,exponent=2):
        '''
        first nengo step, encode function with tuning curves of neurons
        x**exponent
        find_decoders = True -> estimates decoders with pinv
        find_decoders = False -> use already calculated decoders
        '''
        value=np.array([[self.function_exp(x,expo=exponent)] for x in np.linspace(-1,1,nsteps)])
        value=np.reshape(value,nsteps)
        if(find_decoders == True):
            self.decoders = self.find_decoders(value)
        else:
            try:
                self.decoders
            except:
                print 'you never estimated decoders.. please run find_decoders'
                return
        x_estimate_chip = np.dot(self.decoders, [self.tuning_curves])
        self.x_estimate_chip = x_estimate_chip[0]

    def find_decoders_pinv(self,values,neus_in_pop):
        '''
        find optimal linear decoders
        '''
        tuning = self.tuning_curves[neus_in_pop]
        A=np.array(tuning)
        gamma=np.dot(A, A.T) #I diagonal noise add I = eye(len(gamma))
        upsilon=np.dot(A, values)
        ginv=np.linalg.pinv(gamma)
        decoders=np.dot(ginv,upsilon)
        return decoders

    def find_decoders_cvxopt(self, values, neus_in_pop, min_w = 0 , max_w = 1):
        '''
        find optimal decoders using convex bounded optimization
        '''
        tuning = self.tuning_curves[neus_in_pop]
        A=np.array(tuning)
        A_m = np.matrix(np.matrix(A))
        aa = cx.zeros(np.shape(A))
        aa[A_m.nonzero()] = A_m[A_m.nonzero()]
        bb = cx.zeros(np.shape(value))
        bb[value.nonzero()[0]] = value[value.nonzero()[0]]
        m,n = np.shape(aa)
        dec = cx.variable(m)
        p = cx.program(cx.minimize(cx.norm2((aa)*(dec)-(bb))), [cx.leq(dec,max_w),cx.geq(dec,min_w)])
        p.solve()
        return dec.value

    def compute_decoders(self, pop, nsteps= 10, exponent=1, method='pinv'):
        '''
        return decoders for this pop encoding function x**exp 
        method : pinv or cvxopt
        '''
        value=np.array([[self.function_exp(x,expo=exponent)] for x in np.linspace(-1,1,nsteps)])
        value=np.reshape(value,nsteps)
        neus_in_pop  = pop.soma.addr['neu']
        if(method == 'pinv'):
            decoders = self.find_decoders_pinv(value, neus_in_pop)  
        else:
            decoders = self.find_decoders_cvxopt(value, neus_in_pop)

        return decoders

    def plot_encoded_function(self,nsteps=10,input_space=np.linspace(-1,1,10)):
        '''
        run after encoded_function() is called
        '''
        try:
            self.x_estimate_chip
        except:
            print "you must encode a function first.. try nef.encode_function"
            return
        else:
            figure()
            hold(True)
            plot(input_space,self.function_exp(input_space,self.exp),'ro-', label='encoded')
            plot(input_space,self.x_estimate_chip,'bo-', label='decoded')
            rms = self.root_mean_square(self.function_exp(input_space,self.exp),self.x_estimate_chip)
            print 'RMS :', rms
            string_txt = 'RMSE: '+str(rms)
            text(-0.8,1.1,string_txt) 
            xlabel(r'$x$',fontsize=18)
            ylabel(r'$f(x)$',fontsize=18)
            legend(loc='best')
    
    def root_mean_square(self, ideal, measured):
        ''' calculate RMSE
        '''
        return np.sqrt(((ideal - measured) ** 2).mean())

    def transform_function(self, exponent=1, nsteps=10, min_freq = 0 , max_freq = 200, step_dur=500 ):
        '''
        transform function by using tuning curves, encoders information as well as probabilistic weight programmed
        '''
        x_values = np.linspace(min_freq, max_freq, nsteps)
        timebins = np.linspace(0, step_dur*nsteps+1,nsteps+1)
        #we broadcast to all but we only enable input population 
        self.matrix_programmable_broadcast = np.zeros([256,256])
        neu_pop_a = self.pops[0].soma.addr['neu']
        self.matrix_programmable_broadcast[neu_pop_a,0:250] = 1
        self.setup.mapper._program_onchip_broadcast_programmable(self.matrix_programmable_broadcast)
        a = range(256)
        broadcast_syn = self.pop_broadcast[a].synapses['broadcast'][1::512][0]
        for this_phase in range(len(x_values)):
            this_stim = broadcast_syn.spiketrains_regular(x_values[this_phase],t_start=timebins[this_phase],duration=step_dur)
            if(this_phase == 0):
                stim = this_stim
            else:
                stim = pyNCS.pyST.merge_sequencers(stim,this_stim)

        return

    def _fast_ismember(self, A, B):
        '''
        faster ismember implementation only return true false
        A and B are lists
        return tf
        '''
        return [map(lambda val: val in B, a) for a in A]

    def program_weights_prob(self, pop_a, pop_b, weights):
        '''
        it programs synaptic weights with FPGA probabilistic mapper, for onchip (only interface 0) populations
        weights[pre,post]
        '''
        #zero weights no need to be mapped
        source_address = pop_a.soma.paddr 
        nneu_pre = len(source_address)  
        new_value = np.ceil(( (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) ) * (1023 - 0) + 0)
        pre = []
        post = []
        #program weights with prob mapper
        last4bits = 1<<3+0
        #self.setup.mapper._Mappings__open_device()
        self.mem_offset_used = self.mem_offset_mn256r1+256
        final_pres = []
        A = self.pop_broadcast.synapses['programmable'].addr['syntype']
        C = self.pop_broadcast.synapses['programmable'].addr['neu']
        #C = np.reshape(C,[1,len(C)])
        #A = np.reshape(A,[1,len(A)])
        B = self.exc_prog_syn
        E = self.inh_prog_syn            
        syn_e, un = self._ismember(A,B)
        syn_i, un = self._ismember(A,E)
        for this_pre in range(nneu_pre):
            npost = len(weights[this_pre,:])
            pre_address = source_address[this_pre]
            dest_addresses = []
            probs = []
            for this_post in range(npost):
                this_w = weights[this_pre,this_post]
                if(this_w > 0):
                    #connect
                    D = [pop_b.soma.addr['neu'][this_post]]
                    vv, un = self._ismember(self.pop_broadcast.synapses['programmable'].addr['neu'][syn_e],D)
                    dest_address = self.pop_broadcast.synapses['programmable'].paddr[syn_e][vv] - 2**self.bit_input_setupfile
                elif(this_w < 0):
                    D = [pop_b.soma.addr['neu'][this_post]]
                    vv, un = self._ismember(self.pop_broadcast.synapses['programmable'].addr['neu'][syn_i],D)
                    dest_address = self.pop_broadcast.synapses['programmable'].paddr[syn_i][vv] - 2**self. bit_input_setupfile
                #check if there is at least a dest address or if there are connections !=0 
                try:
                    dest_addresses.append(dest_address)
                    final_pres.append(pre_address)
                    probs.append(int(new_value[this_pre,this_post]))
                except:
                    print "neuron pre ", this_pre, " has no connections to post ", this_post
            print 'connecting pre neuron.. ', this_pre
            ### MEMORY OFFSETS
            if(len(dest_addresses) > 1):
                self.mem_offset_used = self.setup.mapper._program_multicast(pre_address, dest_addresses, self.mem_offset_used, self.mem_offset_mn256r1, last4bits, probs, open_device=True)
                pre.append(pre_address)
                post.append(dest_addresses)
        self.mapped_pre = pre
        self.mapped_post = post
        self.probs = probs
        #self.setup.mapper._Mappings__close_device() 
        print "REMEMBER to activate detailed mapping for interface 0 with: nsetup.mapper._program_detail_mapping(2**0)"
    
        return 

    def measure_synaptic_efficacy_prob(self, nsteps=10, freq=120, step_dur=500,nsyn_virt=1, min_w = 0, max_w = +1, exc_col = 255, inh_col=254, do_plot=False):
        '''
        synaptic efficacy with FPGA probabilistic mapper
        '''
        self.syn_eff_prob_step_dur = step_dur
        self.syn_eff_prob_freq = freq 
        self.syn_eff_virt = []
        probs = np.linspace(min_w,max_w,nsteps)
        new_value = np.ceil(( (probs - min_w) / (max_w - min_w) ) * (1023 - 0) + 0)
        #consider encoders for exc and inh synapses
        n_pops = len(self.pops)
        tot_neu_pos = []
        tot_neu_neg = []
        for this_pop in range(n_pops):
            ind_neu_neg = self.encoders[this_pop] < 0
            ind_neu_pos = self.encoders[this_pop] > 0
            id_neu_neg = self.pops[this_pop].soma.addr['neu'][ind_neu_neg]
            id_neu_pos = self.pops[this_pop].soma.addr['neu'][ind_neu_pos]
            tot_neu_pos.append(id_neu_pos)
            tot_neu_neg.append(id_neu_neg)
        a,b = np.shape(tot_neu_pos)
        tot_neu_pos = np.reshape((tot_neu_pos),a*b)
        a,b = np.shape(tot_neu_neg)
        tot_neu_neg = np.reshape((tot_neu_neg),a*b)
        #we make a stimulus that stimulates all neurons via the FPGA mapper with all possible probabilities
        #one synapse per neuron pre
        tf_syn_pos, un = self._ismember(self.pop_broadcast.synapses['programmable'].addr['syntype'], self.exc_prog_syn)
        tf_neu_pos, un = self._ismember(self.pop_broadcast.synapses['programmable'].addr['neu'], tot_neu_pos)
        tf_f_pos = tf_neu_pos & tf_syn_pos
        tf_syn_neg, un = self._ismember(self.pop_broadcast.synapses['programmable'].addr['syntype'], self.inh_prog_syn)
        tf_neu_neg, un = self._ismember(self.pop_broadcast.synapses['programmable'].addr['neu'], tot_neu_neg)
        tf_f_neg = tf_neu_neg & tf_syn_neg
        pre_address_exc = self.pop_broadcast.synapses['virtual_exc'][::4].paddr - 2**self.bit_input_setupfile 
        pre_address_neu_exc = self.pop_broadcast.synapses['virtual_exc'][::4].addr['neu'] 
        #exc
        post_address_pos = self.pop_broadcast.synapses['programmable'].paddr[tf_f_pos] - 2**self.bit_input_setupfile
        post_address_neg = self.pop_broadcast.synapses['programmable'].paddr[tf_f_neg] - 2**self.bit_input_setupfile
        pos_or_neg , un = self._ismember(pre_address_neu_exc, tot_neu_pos)
        #inh
        # 1 << 3 do not broadcast , 0 dest interface shift,
        last4bits = 1<<3+0
        pre = []
        post = []
        syn_stimulus = self.pop_broadcast.synapses['virtual_exc'][::4]
        stim = syn_stimulus.spiketrains_regular(freq,duration=step_dur)
        for this_prob in range(len(probs)):
            self.setup.mapper._Mappings__open_device()
            counter_pos = 0
            counter_neg = 0
            for this_pre in range(len(pre_address_exc)):
                #now we map with probability
                if(pos_or_neg[this_pre]):
                    self.setup.mapper._program_memory(int(self.mem_offset_usb+8+pre_address_exc[this_pre]),int(post_address_pos[counter_pos]),int(last4bits),open_device=False, prob = int(new_value[this_prob]))
                    counter_pos = counter_pos +1
                    pre.append(pre_address_exc[this_pre])
                    post.append(post_address_pos[counter_pos-1])
                else:
                    self.setup.mapper._program_memory(int(self.mem_offset_usb+8+pre_address_exc[this_pre]),int(post_address_neg[counter_neg]),int(last4bits),open_device=False, prob =  int(new_value[this_prob]))
                    counter_neg = counter_neg + 1
                    pre.append(pre_address_exc[this_pre])
                    post.append(post_address_neg[counter_neg-1])
            self.setup.mapper._Mappings__close_device()
            self.setup.mapper._program_detail_mapping(2**self.usb_interface)
            
            #mapper is programmed , now we stimulate
            out = self.setup.stimulate(stim,send_reset_event=False)
            out = out[self.pops[0].soma.channel]
            out.t_start = out.t_stop - step_dur 
            this_prob_firing = out.firing_rate(step_dur)
            self.syn_eff_virt.append(this_prob_firing)
            self.probs = probs

        return  

    def plot_syn_eff_prob(self):
        '''
        plot syn eff with probabilitic weights
        '''
        try:
            self.probs
        except:
            print "you have to measure it first"
            return
        else:
            steps, neu, un = np.shape(self.syn_eff_virt)
            eff = np.array(self.syn_eff_virt)
            dur = self.syn_eff_prob_step_dur / 1000.
            norm_const = self.syn_eff_prob_freq / dur
            for this_neu in range(neu):
                plot(self.probs,eff[:,this_neu]/norm_const,'o-')
                xlabel('steps prob')
                ylabel(r'w [$\theta - H$]')
