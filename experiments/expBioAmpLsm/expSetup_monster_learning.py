#!/usr/env/python

import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('../api/retina/')
sys.path.append('../api/wij/')
sys.path.append('../api/retina/')
sys.path.append('../api/perceptrons/')
sys.path.append('../gui/classification_display/')

from perceptrons import Perceptrons
from wij import SynapsesLearning
from retina import Retina
import time

#FABIO ADD THIS UNDER SVN NOW!!
import missmatch as mm 

######################################
# Configure chip
try:
  is_configured
except NameError:
  print "Configuring chip"
  is_configured = False
else:
  print "Chip is configured: ", is_configured

###########################
# WHAT SHOULD WE DO?
##########################  
plot_planes = False
connect_retina = True
measure_scope = False
read_syn_matrix = False    
do_learning_exp = False    
    
if (is_configured == False):  
    prefix='../'
    setuptype = '../setupfiles/mc_final_mn256r1.xml'
    setupfile = '../setupfiles/final_mn256r1_retina_monster.xml'

    nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
    chip = nsetup.chips['mn256r1']
    nsetup.mapper._init_fpga_mapper()

    p = pyNCS.Population('', '')
    p.populate_all(nsetup, 'mn256r1', 'excitatory')

    #reset multiplexer
    chip.configurator._set_multiplexer(0)
    nsetup.chips['mn256r1'].load_parameters('biases/biases_default.biases')

    #init class synapses learning
    sl = SynapsesLearning(p, 'learning')


    if(plot_planes):
    
        n_trial_per_point = 1 
        tot_ltp = []
        tot_ltd = []
        tot_freqs = []
        #do ltd
        ion()
        pre_f, post_f, wij_inputs, wij_outputs = sl.do_pltp_pltd_vs_fpre_fpost(min_freq = 5, max_freq= 200, nsteps = 3, duration=300, min_inj  = 6e-08, max_inj= 250e-8 , nsteps_injs = 2, init_syn=1)
        sl.plot_plane_ltp_ltd(pre_f, post_f, wij_inputs, wij_outputs, plane = 0)
        for i in range(len(pre_f)):
            directory = 'learning/';
            np.savetxt(directory+"pre_f_"+str(i)+"_ltd.txt", pre_f[i])
            np.savetxt(directory+"post_f_"+str(i)+"_ltd.txt", post_f[i])
            np.savetxt(directory+"wij_inputs_"+str(i)+"_ltd.txt", wij_inputs[i])
            np.savetxt(directory+"wij_outputs_"+str(i)+"_ltd.txt", wij_outputs[i])

        pre_f, post_f, wij_inputs_ltp, wij_outputs_ltp = sl.do_pltp_pltd_vs_fpre_fpost(min_freq = 5, max_freq= 200, nsteps = 3, duration=300, min_inj  = 9e-08, max_inj= 455e-8 , nsteps_injs = 2, init_syn=0)
        sl.plot_plane_ltp_ltd(pre_f, post_f, wij_inputs_ltp, wij_outputs_ltp, plane = 1)
        for i in range(len(pre_f)):
            directory = 'learning/';
            np.savetxt(directory+"pre_f_"+str(i)+"_ltp.txt", pre_f[i])
            np.savetxt(directory+"post_f_"+str(i)+"_ltp.txt", post_f[i])
            np.savetxt(directory+"wij_inputs_"+str(i)+"_ltp.txt", wij_inputs_ltp[i])
            np.savetxt(directory+"wij_outputs_"+str(i)+"_ltp.txt", wij_outputs_ltp[i])
      
    is_configured = True                    
     
    if(connect_retina):

        #neurons
        features_neu = np.linspace(0,127,128)
        perceptron_neu = np.linspace(128,255,128)

        #make on chip network
        feature_pop = pyNCS.Population('', '')
        perceptron_pop = pyNCS.Population('', '')
        feature_pop.populate_by_id(nsetup, 'mn256r1', 'excitatory', features_neu)
        perceptron_pop.populate_by_id(nsetup, 'mn256r1', 'excitatory', perceptron_neu)

        net = Perceptrons(perceptron_pop,feature_pop) 
        net.matrix_learning_pot[:] = 0
        net.upload_config()
        #test
        #syn = feature_pop.synapses['programmable'][::16]
        #stim = syn.spiketrains_poisson(10)
        #nsetup.stimulate(stim,send_reset_event=False)

        #set up filters and connect retina
        inputpop = pyNCS.Population('','')
        inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  
        #reset multiplexer
        chip.configurator._set_multiplexer(0)
        ret = Retina(inputpop)
        ret._init_fpga_mapper()
        pre_teach, post_teach, pre_address, post_address = ret.map_retina_to_mn256r1_randomproj()
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_ret_perceptrons_1.biases')
        
        #two different biases for teacher and inputs
        #matrix_w = np.zeros([256,256])
        #matrix_w[:,0:128]  = 2
        #matrix_w[:,128:256]  = 1
        #nsetup.mapper._program_onchip_programmable_connections(matrix_w)
        
        #retina, pre_address, post_address  = ret.map_retina_to_mn256r1_macro_pixels(syntype='learning')
        #on off retina nsetup.mapper._program_detail_mapping(2**6) on -> 7 
        is_configured = True      
        
    if measure_scope:
        
        #measure trace from the scope for articles
        nsetup.mapper._program_detail_mapping(2**7)
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning.biases')
        import pyAgilent    


        #stimulus learning synapse
        neu = 150
        syn = 12
        index_syn_neu = perceptron_pop.synapses["learning"].addr['neu'] == neu 
        index_syn = perceptron_pop.synapses["learning"].addr['syntype'] == syn 
        index_tot = index_syn & index_syn_neu
        this_syn = perceptron_pop.synapses["learning"][index_tot] 
        stim = this_syn.spiketrains_poisson(165, duration=650)
        
        #stimulus virtual synapse post neuron
        index_syn_neu_virt = perceptron_pop.synapses["virtual_exc"].addr['neu'] == neu
        syn_virt = perceptron_pop.synapses["virtual_exc"][index_syn_neu_virt]
        stim_v = syn_virt.spiketrains_poisson(100, duration=650)
        final_stim = pyNCS.pyST.merge_sequencers(stim_v, stim)

        nsetup.stimulate(final_stim,send_reset_event=False)
             
        #
        osc_a = pyAgilent.Agilent(host="172.19.10.159");
        osc_a._send_command('WAV:FORM RAW');
        osc_b = pyAgilent.Agilent(host="172.19.10.156");
        osc_b._send_command('WAV:FORM RAW');       
        
        membrane = osc_b._read_data_from_channel(2)
        weight = osc_b._read_data_from_channel(1)
        pre = stim[1].raw_data()[:,0]#osc_b._read_data_from_channel(3)
        up = osc_a._read_data_from_channel(2)
        dn = osc_a._read_data_from_channel(3)
        calc = osc_b._read_data_from_channel(4)
        time = np.linspace(0,1000, len(membrane))
        creq = osc_a._read_data_from_channel(4)
        post_exc = stim_v[1].raw_data()[:,0]
        
        figure()
        subplot(6,1,1)
        plot(time,membrane, label='membrane')
        ylabel('V')
        legend(loc='best')
        subplot(6,1,2)
        plot(time,weight, label='weight')
        ylabel('V')
        legend(loc='best')
        subplot(6,1,3)
        plot(time,calc, label='X (calc)')
        ylabel('V')
        legend(loc='best')
        subplot(6,1,4)
        plot(time,up, label='Up')
        ylabel('V')
        legend(loc='best')
        subplot(6,1,5)
        plot(time,dn, label='Dn')
        ylabel('V')
        legend(loc='best')
        subplot(6,1,6)
        plot(pre, np.linspace(1,1,len(pre)),  'o',  label='pre')
        ylabel('V')
        legend(loc='best')
        xlabel('time (ms)')

        
        np.savetxt("frontiers/membrane_f_0.txt",membrane)
        np.savetxt("frontiers/weight_f_0.txt",weight)
        np.savetxt("frontiers/calc_f_0.txt",calc)
        np.savetxt("frontiers/up_f_0.txt",up)
        np.savetxt("frontiers/dn_f_0.txt",dn)
        np.savetxt("frontiers/time_f_0.txt",time)
        np.savetxt("frontiers/pre_f_0.txt", pre)
        np.savetxt("frontiers/chipreq_f_0.txt", creq)
        np.savetxt("frontiers/post_f_0.txt", post_exc)
        
        osc_b.read_data([2,1])
        osc_b._labels = {1: 'Weight', 2: 'Vmem'}
        osc_a.read_data([1,2,3,4])
        osc_a._labels = {1: 'Calcium', 2: 'up', 3: 'down', 4: 'pre spike'}
        osc_a.plot_all_data()
        figure()
        osc_b.plot_all_data()
        
    def learn(n_presentations):
        '''
        n_presentations -> ntimes presentation of the trainin set
        n_reads -> number of reads of the synaptic matrix
        '''
        
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_ret_perceptrons_1.biases')
        chip.configurator.set_parameter("PDPI_TAU_P", 2.4e-05)
        chip.configurator.set_parameter("PDPI_THR_P", 0.0)
        chip.configurator.set_parameter("VDPIE_THR_P", 0.0)
        chip.configurator.set_parameter("VDPIE_TAU_P", 2.0e-05)
        chip.configurator.set_parameter("VDPII_TAU_N", 2.0e-05)
        chip.configurator.set_parameter("VDPII_THR_N", 0.0)
        
        win.train(xi,yi,n_presentations)   
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijtesting_ret_perceptrons.biases')

           
           
    def learn_and_save_all_matrices(n_presentations)  :     
        '''
        Hamming distance plot
        '''
        matrices = []
        for i in range(n_presentations):
            nsetup.chips['mn256r1'].load_parameters('biases/biases_wijlearning_ret_perceptrons_1.biases')
            chip.configurator.set_parameter("PDPI_TAU_P", 2.4e-05)
            chip.configurator.set_parameter("PDPI_THR_P", 0.0)
            chip.configurator.set_parameter("VDPIE_THR_P", 0.0)
            chip.configurator.set_parameter("VDPIE_TAU_P", 2.0e-05)
            chip.configurator.set_parameter("VDPII_TAU_N", 2.0e-05)
            chip.configurator.set_parameter("VDPII_THR_N", 0.0)
            
            win.train(xi,yi,1)   
            nsetup.chips['mn256r1'].load_parameters('biases/biases_wijtesting_ret_perceptrons.biases')
            read_syn_matrix_learning()
            net.upload_config(matrix_learning_state=False, matrix_exc_inh=True, matrix_programmable_w=False, matrix_programmable_rec=False, matrix_learning_rec=True)
            matrices.append(sl.state)
        return matrices
            
    def read_syn_matrix_learning():
                  
        matrix_rec = np.zeros([256,256])
        nsetup.mapper._program_onchip_plastic_connections(matrix_rec)
        nsetup.mapper._program_onchip_programmable_connections(matrix_rec)
        time.sleep(0.2)
        sl.get_br()
        time.sleep(0.2)

        #net.upload_config(matrix_learning_state = False, matrix_programmable_w = False )
        
        #chip.configurator.set_parameter("PDPI_TAU_P", 2.4e-05)
        #chip.configurator.set_parameter("PDPI_THR_P", 0.0)
        #chip.configurator.set_parameter("VDPIE_THR_P", 0.0)
        #chip.configurator.set_parameter("VDPIE_TAU_P", 2.0e-05)
        #chip.configurator.set_parameter("VDPII_TAU_N", 2.0e-05)
        #chip.configurator.set_parameter("VDPII_THR_N", 0.0)
            

        #switch on retina input    
        #nsetup.mapper._program_detail_mapping(2**6)
        
    def hamdist(a, b):
        """Count the # of differences between equal length strings str1 and str2"""
            
        return np.sum(np.abs(a-b))      
        

    def test_and_score():
        ##########################################
        nsetup.chips['mn256r1'].load_parameters('biases/biases_wijtesting_ret_perceptrons.biases')
        print win.test(xi,yi)  
          

if(do_learning_exp):        
    ##########################################
    # UPLOAD CONFIG ON NEUROMORPHIC CORE
    ##########################################
    net.upload_config()

    ###########################################
    # LOAD DATABASE AND OPEN WINDOW FOR STIMULI
    ##########################################
    nsetup.mapper._program_detail_mapping(2**7)
    import display_datasets as dd
    my_data_folder = '/home/federico/Documents/ini/code/python/mn256r1_ncs/api/missmatch/data/'
    datasets = mm.seek_data(my_data_folder)
    x, y = mm.load_data(my_data_folder + "caltech101_car_sideVSmotorbikes.csv") 
    label_true = 4#2#4
    label_false = 20#72#20
    y_true = np.where(y == label_true, 1, 0)
    y_false = np.where(y == label_false, -1, 0)
    y_select = np.nonzero(y_true + y_false)[0]
    y = (y_true + y_false)[y_select]
    x = x[:, y_select]
    xi, yi, xt, yt = mm.sample_train_test(x, y, train_fraction=0.05,
                                          normalization=None)
    # pattern_shape is the shape of the original images
    win = dd.mainWindow(setup=nsetup, net = net,  pattern_shape=(128, 128),
                     show_shape=(360, 360),
                     roll_fac=3,
                     normalization=None, speed_fac=5)
                     
    ##########################################
    # L E A R N !!!
    ##########################################
    #this_w = learn(1) 
    matrici = learn_and_save_all_matrices(1) #epoch
    ham_tot = []
    for i in range(len(matrici)-1):
        this_ham = hamdist(matrici[i].reshape(1,256*256)[0], matrici[i+1].reshape(1,256*256)[0])
        ham_tot.append(this_ham)
    plot(ham_tot, 'o-')
    ylabel('Hamming Distance')
    xlabel('nsteps')    

    ##########################################
    # T E S T and S C O R E
    ##########################################
    nsetup.chips['mn256r1'].load_parameters('biases/biases_wijtesting_ret_perceptrons.biases')
    print win.test(xi,yi)  
    #test_and_score() 
        

