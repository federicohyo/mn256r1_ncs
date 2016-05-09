# This configurator uses information of the XML to set internal variables. This
# feature is currently not supported by pyNCS, which instead first create a
# Configurator object and then parses NHML to just fill it up with dictionaries.
# Hence, we have to overwrite the __parseNHML__ function so that also internal
# variables are set while parsing the NHML. This function uses etree from the
# lxml module, that is why we import it here, as well as Parameter from pyNCS
import numpy as np
from lxml import etree
import time,os,sys

from pyNCS.api.ConfAPI import ConfiguratorBase, MappingsBase, Parameter

from client_monster import Client

class Configurator(ConfiguratorBase):
    def __init__(self, bias_folder='biases', host='localhost', port=50019):
        '''
        '''
        self._scales = []
        self._client = Client(host, int(port))
        super(Configurator, self).__init__()

    def add_parameter(self, param):
        # for some reason, this function gets called twice *per parameter*
        # (by pyNCS...!)
        if 'Scale' in param.get('SignalName'):
            if not self._coarse2val(param.get('Value')) in self._scales:
                self._scales.append(self._coarse2val(param.get('Value')))
        else:
            parameter = Parameter({'SignalName': ''}, self)
            parameter.__parseNHML__(param)
            self.parameters[parameter.SignalName] = parameter

    #def getValue(self, param_name):
        #print "Sorry, can't get parameter from biasgen..."
        #raise NotImplementedError

    def _coarse2val(self, coarse_string):
        """
        >>> _coarse2val('3.2u')
        3.2e-6
        """
        if 'p' in coarse_string:
            exponent = 1e-12
        elif 'n' in coarse_string:
            exponent = 1e-9
        elif 'u' in coarse_string:
            exponent = 1e-6
        val = float(coarse_string[:-1])
        return val * exponent

    def _coarse2scale(self, coarse_string):
        """
        returns int
        """
        val = self._coarse2val(coarse_string)
        return np.argmin(abs(np.r_[self._scales] - val))

    def _coarsify(self, bias_value):
        which_scale = np.max(np.argwhere((\
            np.r_[self._scales] >= bias_value)).flatten())
        fines = np.linspace(0, self._scales[which_scale], 256)
        which_fine = np.argmin(np.abs(fines - bias_value))
        actual_value = fines[which_fine]
        return which_scale, which_fine, actual_value

    def _convert_bias(self, biasName, coarse, fine):
        #adddes
        param = self.get_parameters()[biasName]
        addr = int(param.param_data['Latch'])
        biasLowHigh = 1 if "High" in param.param_data['HighLow'] else 0
        biasCascode = 1 if "Normal" in param.param_data['BiasType'] else 0
        biasType = 1 if "NBias" in param.param_data['FET'] else 0
        biasEnable = 1 if "Enable" in param.param_data['Enable'] else 0
        #configuration bits
        confBits = biasLowHigh << 3 | biasCascode << 2 | biasType << 1 | biasEnable
        #ssn and ssp
        if addr == 65 or addr == 66: 
            inbits = addr << 16 | 63 << 10 | fine << 4 | confBits
        else:
            special = 0 #???
            #bit pattern
            coarse_reversed = sum(1 << (2 - i) for i in range(3) if coarse >> i & 1)
            inbits = addr << 16 | special << 15 | coarse_reversed << 12 | fine << 4 | confBits
        return inbits

    def set_parameter(self, param_name, param_value):
        param = self.get_parameters()[param_name]
        addr = int(param.param_data['Latch'])
        if(addr == 65 or addr == 66):
            in_bits = self._convert_bias(param_name, 0, int(param_value))
        else:
            if param_value > self._scales[0]:
                print "Warning: value %1.2e for %s is out of range."%\
                        (param_value, param_name)
                param_value = self._scales[0]
            which_coarse, which_fine, actual_value =\
                    self._coarsify(param_value)
            in_bits = self._convert_bias(param_name, which_coarse, which_fine)
            print_bits = '{:23b}'.format(in_bits)
            value_coarse = str(self._scales[which_coarse])
            value_fine = str(self._scales[which_coarse] / 256. * which_fine)
            bias_lo_hi = str(1 if 'High' in\
                             self.parameters[param_name].param_data['HighLow']\
                             else 0)
            bias_cascode = str(1 if 'Normal' in\
                               self.parameters[param_name].param_data['BiasType']\
                               else 0)
            bias_type = str(1 if 'N' in\
                            self.parameters[param_name].param_data['FET']\
                            else 0)
            bias_enable = str(1 if 'Enable' in\
                              self.parameters[param_name].param_data['Enable']\
                              else 0)
            #print param_name + ", " + value_coarse + " (" + str(value_fine) +\
                    #")" + ", " + bias_lo_hi + ", " + bias_cascode + ", " +\
                    #bias_type + ", " + bias_enable + " -> " + print_bits[0:7] +\
                    #' ' + print_bits[7:24]
        self._send_32(in_bits)
    
    def _send_32(self, in_bits, debug=False):
        programming_bits = np.bitwise_and(in_bits,65535) #bits 0 16 
        address_branch = (np.bitwise_and(in_bits,8323072)>>16) #bits 17 to 22 
        final_address = (programming_bits<<7) + (address_branch) + 2**31
        if debug:
            print "in_bits", in_bits
            print "send stuff"
            print "address_branch", np.binary_repr(address_branch)
            print "programming_bits", np.binary_repr(programming_bits)
            print "address_branch", (address_branch)
            print "programming_bits", (programming_bits<<7)
            print "final address", final_address
        self._client.send(str([0,final_address]))
        time.sleep(0.001)

    def _init_chip(self):
        for k, v in self.get_parameters().iteritems():
            try:
                in_bits = self._convert_bias(k, self._coarse2scale(v.param_data['DefaultCoarse']),
                                             int(v.param_data['DefaultFine']))
                print k, v.param_data['DefaultCoarse'], int(v.param_data['DefaultFine']), in_bits
                self._send_32(in_bits)
                time.sleep(0.1)
            except KeyError:
                pass

    def _set_monitor_neuron(self, neuron_id):
        """
        routes the Vmem of neuron_id to monitor pad
        """
        self._client.send(str([0,1249024]))
        time.sleep(0.001)
        self._client.send(str([0,1249025]))
        time.sleep(0.001)
        self._client.send(str([0,neuron_id + 1249284])) #PLEASE FIX!
        time.sleep(0.001)
        self._client.send(str([0,1249025]))
        time.sleep(0.001)
        self._client.send(str([0,neuron_id + 1249284]))
        time.sleep(0.001)
    
    def _set_neuron_tau1(self, neurons):
        """
        set neurons to tau 1 ; neurons is an array 
        """
        for i in range(len(neurons)):
            self._client.send(str([0,int(neurons[i]+1249026)]))
            time.sleep(0.001)
            self._client.send(str([0,1249283]))     
            time.sleep(0.001)               
            self._client.send(str([0,int(neurons[i]+1249026)]))
            time.sleep(0.001)

    def _set_all_neu_tau2(self):
        """
        set all neuron to tau 2 (not possible to set single ones)
        """
        self._client.send(str([0,1249282]))
        time.sleep(0.001)
        self._client.send(str([0,1249283]))
        time.sleep(0.001)

    def _set_multiplexer(self,mux_value):
        """
        set synaptic multiplexer to merge line of synapses 
        """
        self._client.send(str([0,1249542])) ## we need to reset to zero.. ??
        time.sleep(0.001)
        self._client.send(str([0,int(mux_value+1249542)]))
        time.sleep(0.001)

class Mappings(MappingsBase):
    def __init__(self,host='localhost', port=50019, use_device = True, time_sleep=0.006, force_clear = True):
        '''
        MappingsBase()
        Base class for managing mappinger of raggedstone 2 board

        Contains methods:
        - add_mappings() (required)
        - set_mappings(mappings) (optional)
        - get_mappings()
        - clear_mappings()
        - del_mappings() (optional, not used by pyNCS by default)
        '''
        MappingsBase.__init__(self)
        self._client = Client(host, int(port))
        #### TODO: these parameters need to be loaded from the XML file setupfile
        self.mem_offset_mn256r1 = 500
        self.mem_offset_usb = 200
        self.in_bit = 22 #defined in the xml setupfile
        self.possible_interfaces = 8 
        self.usb_interface = 1
        self.mn256r1_interface = 0
        #### end parameters
        self.time_sleep = time_sleep
        self.force_clear = force_clear
        self.use_device = use_device
        self.device = '/dev/aerfx2_0'
        self.state_device_open = False
        self.fd = []

    def _init_fpga_mapper(self):
        ####
        print "1) clear mapper registers"
        #clear registers
        for i in range(self.possible_interfaces):
            self._program_address_range(int(i),(2**32)-1)
            self._program_offset(int(i),int(0))
            self._program_bulk_spec(int(i),0)
        self._program_detail_mapping(0)
        #clear memory
        if self.force_clear:
            print "2) clear mapper memory"
            self.clear_mappings()
        else:
            print "2) warning: not cleaning mapper memory, we assume it is empty"
        print "3) program bulk specifications"
        #default bulk specs chip->usb and usb->chip
        self._program_bulk_spec(self.usb_interface,2**self.mn256r1_interface)
        self._program_bulk_spec(self.mn256r1_interface,2**self.usb_interface)
        #now we subtract the bit from the usb events defined in the setupfile
        self._program_offset(self.usb_interface,self.mem_offset_usb)
        self._program_memory(self.mem_offset_usb+0, 0, 0)
        self._program_offset(self.mn256r1_interface, self.mem_offset_mn256r1)
        self._program_memory(self.mem_offset_mn256r1+0, 0, 0)
        #program address ranges
        self._program_address_range(self.usb_interface,(2**32)-1)
        self._program_address_range(self.mn256r1_interface,(2**32)-1)
        print "4) clear onchip recurrent connections"
        matrix = np.zeros([256,512])
        self._program_onchip_recurrent(matrix)
        print "##### mapper initialization completed"

    def _program_memory_range(self, start, stop, value, last4bits):
        '''
        this function program FPGA mapper last4 bits 
        '''
        addresses = np.arange(start,stop)
        writebuf_0 = int(0)
        writebuf_1 = int(0)
        if(self.use_device):
            self.__open_device()
        for i in range(len(addresses)):
            writebuf_1 |= 1<<30
            writebuf_1 |= int(last4bits);
            writebuf_1 |= i<<4
            writebuf_0 = int(value)
            if(self.use_device==False):
                self._client.send(str([writebuf_0,writebuf_1]))
                time.sleep(self.time_sleep)
            else:
                self.__write_to_device(str([writebuf_0,writebuf_1]))
        if(self.use_device):
            self.__close_device()

    def _program_memory(self, address, value, last4bits, open_device=True, prob=1023):
        '''
        this function program FPGA mapper memory
        '''
        writebuf_0 = int(0)
        writebuf_1 = int(0)
        writebuf_1 |= 1<<30
        writebuf_1 |= int(last4bits);
        writebuf_1 |= address<<4
        writebuf_0 |= int(value)
        writebuf_0 |= prob<<22
        if(self.use_device):
            if(open_device == True):
                self.__open_device()
            self.__write_to_device(str([writebuf_0,writebuf_1]))
            if(open_device == True):
                self.__close_device()
        else:
            self._client.send(str([writebuf_0,writebuf_1]))
            time.sleep(self.time_sleep)

    def _program_multicast(self, pre, post, memoffset, memoffset_interface, last4bits, prob, open_device=False):
        '''
        program one to many mapper with probability prob [0,1023] 
        prob is vector of len(post)
        '''
        if(np.size(pre) > 1):
            print 'this function map only one pre to multiple post neurons'
            return
        if(len(post) < 2):
            print 'this function map only one pre to multiple post neurons'
            return
        memory_loc_one_to_many = memoffset
        fan_out = len(post)
        if(open_device):
            self._Mappings__open_device()
        self._program_memory(int(memory_loc_one_to_many),int(fan_out),0,prob=0) #zero for multicast
        counter = 1
        for this_post in range(0,fan_out):
            self._program_memory(int(memory_loc_one_to_many+counter),int(post[this_post]),last4bits,prob=prob[this_post],open_device=True)
            counter = counter + 1
        self._program_memory(int(memoffset_interface+8+pre),int(memory_loc_one_to_many),0,prob=0,open_device=True)
        memorypointer = memory_loc_one_to_many+counter

        #if(open_device):
        #    self._Mappings__close_device()

        return memorypointer

    def _program_bulk_spec(self, interface_num, bulk_mask):
        '''
        program FPGA bulk mapping. ie. map entire inferface to another one
        '''
        writebuf_0 = int(0)
        writebuf_1 = int(0)
        writebuf_1 |= 1<<30
        writebuf_1 |= 1<<29
        writebuf_1 |= int(interface_num)
        writebuf_0 = int(bulk_mask)
        w_tot = (writebuf_1 << 32) + writebuf_0
        if(self.use_device):
            self.__open_device()
            self.__write_to_device(str([writebuf_0,writebuf_1]))
            self.__close_device()
        else:
            self._client.send(str([writebuf_0,writebuf_1]))
            time.sleep(self.time_sleep)

    def _program_detail_mapping(self, mapping_mask):
        '''
        enable detail mapping of interface
        '''
        writebuf_0 = int(0)
        writebuf_1 = int(0)
        writebuf_1 |= 1<<30
        writebuf_1 |= 1<<29
        writebuf_1 |= 1<<28
        writebuf_0 = int(mapping_mask)
        if(self.use_device):
            self.__open_device()
            self.__write_to_device(str([writebuf_0,writebuf_1]))
            self.__close_device()
        else:
            self._client.send(str([writebuf_0, writebuf_1]))
            time.sleep(self.time_sleep)

    def _program_address_range(self, interface_num, address_range):
        '''
        memory address range for interface num
        '''
        writebuf_0 = int(0)
        writebuf_1 = int(0)
        writebuf_1 |= 1<<30
        writebuf_1 |= 1<<31
        writebuf_1 |= interface_num
        writebuf_0 = int(address_range)
        if(self.use_device):
            self.__open_device()
            self.__write_to_device(str([writebuf_0,writebuf_1]))
            self.__close_device()
        else:
            self._client.send(str([writebuf_0,writebuf_1]))
            time.sleep(self.time_sleep)

    def _program_offset(self, interface_num, offset):
        '''
        program offset for interface
        '''
        writebuf_0 = int(0)
        writebuf_1 = int(0)
        writebuf_1 |= 1<<30
        writebuf_1 |= 1<<28
        writebuf_1 |= interface_num
        writebuf_0 = int(offset)
        if(self.use_device):
            self.__open_device()
            self.__write_to_device(str([writebuf_0,writebuf_1]))
            self.__close_device()
        else: 
            self._client.send(str([writebuf_0,writebuf_1]))
            time.sleep(self.time_sleep)

    def _set_recurrent_bit(self, neu, syn, onoff):
        '''
        Set recurrent bit of row (neu) and col (syn) to on onoff (1 or 0)
        '''
        plastic = False
        programmable = False
        if (syn < 256):
            plastic = True
        else:
            programmable = True
        if(plastic):
            offset = 788992
            matrix_value = onoff
            value = matrix_value*256*256+syn*256+neu+offset
        if(programmable):
            offset = 133632
            matrix_value = onoff
            value = matrix_value*256*256+(syn-256)*256+neu+offset
        if(self.use_device):
            self.__write_to_device(str([0,int(value)]))
        else:
            self._client_send(str([0,int(value)]))
            time.sleep(self.time.sleep)

    def __open_device(self):
        self.fd = os.open(self.device, os.O_WRONLY | os.O_NONBLOCK)
        self.state_device_open = True
 
    def __close_device(self, force = False):
        if(self.state_device_open == True or force == True):
            os.close(self.fd)

    def __write_to_device(self, data):
        if(self.state_device_open):
            data = data.strip("[]")
            data = np.array(data.split(","), dtype="uint32")
            to_write = np.r_[int(data[0]), int(data[1])].astype('uint32').tostring()
            os.write(self.fd, to_write)
        else:
            self.__open_device(self)
            data = data.strip("[]")
            data = np.array(data.split(","), dtype="uint32")
            to_write = np.r_[int(data[0]), int(data[1])].astype('uint32').tostring()
            os.write(self.fd, to_write)
            self.__close_device(self)

    def _program_onchip_recurrent(self, matrix):
        '''
        program recurrent matrix of both plastic and programmable synapses [shape(256,512) 256-> neu ,  512->syn]
        '''
        if( np.shape(matrix) != (256,512) ):
            print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,512)'
            return 
        else:
            val = 0
            updated_row = 0
            if(self.use_device):
                self.__open_device()
            for col in range(512):
                for row in range(256):
                    if(updated_row <= val):
                        val = int((col/512.0)*100.0)
                        #print '\r[{0}] {1}%'.format('#'*(val/10), val)
                        updated_row = val
                    self._set_recurrent_bit(row,col,matrix[row,col])
            if(self.use_device):
                self.__close_device()

    def _program_onchip_broadcast_programmable(self,matrix):
        '''
        program broadcast synapses for programmable matrix shape([256,256])
        state = 1 -> broadcast on
        state = 0 -> broadcast off
        '''
        if( np.shape(matrix) != (256,256) ):
            print 'matrix has wrong size.. please input a binary matrix (np.array) with shape  = (256,256)'
            return
        else:
            val = 0
            updated_row = 0
            if(self.use_device):
                self.__open_device()
            for col in range(256):
                for row in range(256):
                    self._set_non_plastic_broadcast(row,col,matrix[row,col])
            if(self.use_device):
                self.__close_device()

    def _program_onchip_broadcast_learning(self,matrix):
        '''
        program broadcast synapses for learning matrix shape([256,256])
        state = 1 -> broadcast on
        state = 0 -> broadcast off
        '''
        if( np.shape(matrix) != (256,256) ):
            print 'matrix has wrong size.. please input a binary matrix (np.array) with shape  = (256,256)'
            return
        else:
            val = 0
            updated_row = 0
            if(self.use_device):
                self.__open_device()
            for col in range(256):
                for row in range(256):
                    self._set_plastic_broadcast(row,col,matrix[row,col])
            if(self.use_device):
                self.__close_device()

    def _set_non_plastic_broadcast(self,neu,syn,state):
        '''
        set single programmable synapse to be broadcast
        state = 1 -> broadcast on
        state = 0 -> broadcast off
        '''
        value = 4*2*256*256+state*256*256+syn*256+neu+133632
        to_write = np.r_[int(0), int(value)].astype('uint32').tostring()
        if(self.use_device):
            if(self.state_device_open):
                os.write(self.fd, to_write)
            else:
                self.__open_device(self)
                os.write(self.fd, to_write)
                self.__close_device(self)

    def _set_plastic_broadcast(self,neu,syn,state):
        '''
        set single plastic synapse to be broadcast
        state = 1 -> broadcast on
        state = 0 -> broadcast off
        '''
        value = 1*2*256*256+state*256*256+syn*256+neu++788992
        to_write = np.r_[int(0), int(value)].astype('uint32').tostring()
        if(self.use_device):
            if(self.state_device_open):
                os.write(self.fd, to_write)
            else:
                self.__open_device(self)
                os.write(self.fd, to_write)
                self.__close_device(self)

    def _set_weight_single_syn_programmable(self, neu, syn, w):
        '''
        set single programmable synapse to weight w
        '''
        if(w> 3):
            print "error weights!! please enter w in range [0,3]"
            raise Exception
        if w == 0:
            bit0=0
            bit1=0
        if w == 1:
            bit0=1
            bit1=0
        if w == 2:
            bit0=0
            bit1=1
        if w == 3:
            bit0=1
            bit1=1
        value = 2*2*256*256+bit0*256*256+syn*256+neu+133632
        value1 = 3*2*256*256+bit1*256*256+syn*256+neu+133632
        if(self.use_device):
            self.__write_to_device(str([0,int(value)]))
            self.__write_to_device(str([0,int(value1)]))
        else:
            self._client_send(str([0,int(value)]))
            time.sleep(self.time.sleep)
            self._client_send(str([0,int(value1)]))
            time.sleep(self.time.sleep)

    def _program_onchip_weight_matrix_programmable(self, matrix):
        '''
        program weight for programmable synapses
        '''
        if( np.shape(matrix) != (256,256) ):
            print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
            return 0
        else:
            if(self.use_device):
                self.__open_device()
            for neu in range(256):
                for syn in range(256):
                    self._set_weight_single_syn_programmable(neu,syn,matrix[neu,syn])
            if(self.use_device):
                self.__close_device()
    
    def _set_syn_exc_inh(self, neu, syn, ei):
        '''
        set single programmable synapse to be exc or inh
        '''
        value = 1*2*256*256+ei*256*256+syn*256+neu+133632
        if(self.use_device):
            self.__write_to_device(str([0,int(value)]))
        else:
            self._client_send(str([0,int(value)]))
            time.sleep(self.time.sleep)

    def _program_onchip_exc_inh(self, matrix):
        '''
        this function set the programmable synapses to be exc or inhibitory input np.shape(256,256) | 0 = inh / 1 = exc
        '''
        if( np.shape(matrix) != (256,256) ):
            print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
            raise Exception 
        else:
            if(self.use_device):
                self.__open_device()
            for neu in range(256):
                for syn in range(256):
                    self._set_syn_exc_inh(neu,syn,matrix[neu,syn])
            if(self.use_device):
                self.__close_device()

    def _program_onchip_plastic_connections(self, matrix):
        '''
        program connections matrix for learning synapses
        '''
        if( np.shape(matrix) != (256,256) ):
            print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
            raise Exception
        else:
            if(self.use_device):
                self.__open_device()
            for col in range(256):
                for row in range(256):
                    self._set_recurrent_bit(row,col,matrix[row,col])
            if(self.use_device):
                self.__close_device()

    def _program_onchip_programmable_connections(self, matrix):
        '''
        program connection matrix for programmable synapses
        '''
        if( np.shape(matrix) != (256,256) ):
            print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
            raise Exception
        else:
            if(self.use_device):
                self.__open_device()
            for col in range(256,512):
                for row in range(256):
                    self._set_recurrent_bit(row,col,matrix[row,col-256])
            if(self.use_device):
                self.__close_device()

    def _set_syn_plastic_state(self, neu, syn, state):
        '''
        Set plastic weight to value (1/0) of row (neu) and column (synapse) 
        '''
        if(state == 1):
            vv = 1*256*256+syn*256+neu+1051136
        elif(state == 0):
            vv = syn*256+neu+1051136
        if(self.use_device):
            self.__write_to_device(str([0,int(vv)]))
        else:
            self._client_send(str([0,int(vv)]))
            time.sleep(self.time_sleep)

    def _program_onchip_learning_state(self, matrix):
        '''
        set state plastic synapses (only binary)
        '''
        if( np.shape(matrix) != (256,256) ):
            print 'matrix has wrong size.. please input a binary matrix (np.array) with shape = (256,256)'
            return 0
        else:
            if(self.use_device):
                self.__open_device()
            for row in range(256):
                for col in range(256):
                   self._set_syn_plastic_state(row,col,matrix[row,col])
            if(self.use_device):
                self.__close_device()

    def add_mappings(self, mappings):
        #IMPLEMENT (REQUIRED)
        '''
        Adds *mappings* to the mappings table.

        Inputs:
        *mappings*: a two-dimenstional iterable
        '''
        pass

    def get_mappings(self):
        #IMPLEMENT (REQUIRED)
        '''
        Returns an array representing the mappings
        '''
        return None

    def clear_mappings(self):
        #IMPLEMENT (REQUIRED)
        '''
        Clears the mapping table. No inputs
        '''
        self._program_memory_range(0,(2**10)-1, 0, 0)
        return None

    def set_mappings(self, mappings):
        #CONVIENCE FUNCTION, IMPLEMENTATION NOT REQUIRED
        '''
        Clears the mapping table and adds *mappings* to the mappings table.

        Inputs:
        *mappings*: a two-dimenstional iterable
        '''
        self.clear_mappings()
        self.add_mappings(mappings)

    def del_mappings(self):
        #IMPLEMENT (OPTIONAL)
        '''
        Clears the mapping table. No inputs
        '''
        raise NotImplementedError('del_mappings has not been implemented')

