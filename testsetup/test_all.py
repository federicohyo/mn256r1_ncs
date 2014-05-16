# === TEST FUNCTIONS OF MN256R1 =================================================
# === federico@ini.phys.ethz.ch
# ===============================================================================

###################
# Import libraries
###################
import pyNCS
import os as os
import math
import numpy as np
from pylab import *
import time
import pyNCS
from pyNCS.neurosetup import NeuroSetup

import sys
# no need to change these library files
sys.path.append('_utils/')
sys.path.append('_utils/mapperlib')
sys.path.append('_utils/biasgenlib')

# here you need to confiture your mapper
sys.path.append('mapper/') 

import biasusb_wrap
import mapper_wrap
import mapper
import AERmn256r1
import MN256R1Configurator

failed = False

#init mappings
offsetfirstchip = 500
offset_usb = 1000
offsetadcs = 2000
print "Init mapper ..."
mapper.init_mappings(offsetfirstchip,offset_usb,offsetadcs)

#we set the default biases reset latches and so on
biasconf = MN256R1Configurator.MN256R1Configurator()
default_biases = biasconf.load_default()
biasconf.set_all_biases(default_biases)

#################################################
#   CONFIGURE default synaptic matrix
#################################################

import conf_mn256r1

print "Program connections weights ..."
AERmn256r1.set_weight_matrix_programmable(conf_mn256r1.matrix_programmable_w)
time.sleep(1)
print "Program connections exc/inh ..."
AERmn256r1.set_matrix_exc_inh(conf_mn256r1.matrix_programmable_exc_inh)
time.sleep(1)
print "Program recurrent plastic connections ..."
AERmn256r1.set_connections_matrix_plastic(conf_mn256r1.matrix_learning_rec)
time.sleep(1)
print "Program connections ..."
AERmn256r1.set_connections_matrix_programmable(conf_mn256r1.matrix_programmable_rec)
time.sleep(1)
print "Program plastic weights ..."
AERmn256r1.set_weight_matrix_plastic(conf_mn256r1.matrix_learning_pot)
time.sleep(1)


#################################################
#   PYNCS setup
#################################################
#chech if we can receive spikes from the device
print 'setting up pyNCS setup'
import pyNCS
from pyNCS.neurosetup import NeuroSetup

# Define file paths
prefix='./chipfiles/'
setuptype = 'setupfiles/mc_final_mn256r1.xml'
setupfile = 'setupfiles/final_mn256r1.xml'

#Create Chip objects
setup = pyNCS.NeuroSetup(setuptype,setupfile,prefix=prefix)
chip = setup.chips['mn256r1']

neu_to_stim = [np.random.randint(256) for i in range(16)]
neu_to_stim = np.sort(np.unique(neu_to_stim))
neu_back_ground = np.linspace(0,255,256)
neu_back_ground = np.delete(neu_back_ground,neu_to_stim)

#all neurons
all_neu = pyNCS.Population('neurons', 'for fun')
all_neu.populate_by_id(setup,'mn256r1','excitatory',np.linspace(0,255,256))

stim_neu = pyNCS.Population('neurons', 'for fun')
stim_neu.populate_by_id(setup,'mn256r1','excitatory',neu_to_stim)

back_neu = pyNCS.Population('neurons', 'for fun')
back_neu.populate_by_id(setup,'mn256r1','excitatory',neu_back_ground)

#switch on monitors
mon_all = pyNCS.monitors.SpikeMonitor(all_neu.soma)
mon_stim = pyNCS.monitors.SpikeMonitor(stim_neu.soma)
mon_back = pyNCS.monitors.SpikeMonitor(back_neu.soma)
setup.monitors.import_monitors([mon_all,mon_stim,mon_back])

#======================================================================
#= raggedstone 2 only send stuff when buffer is full
#= we use neuron 255 to be set to tau2 so that it will fire like crazy
#######################################################################

#====================================listen to the silicon
setup.stimulate({},send_reset_event=False,duration=1000)
pyNCS.monitors.RasterPlot(mon_all)
x_start = np.min(mon_all.sl.raw_data()[:,0])
x_stop = np.max(mon_all.sl.raw_data()[:,0])
xlim([x_start,x_stop])


#======================================================================
print 'set neuron 255 to tau 1 to zero and its refractory to zero, it will be our clock neuron...'
AERmn256r1.set_neuron_tau_1([255])
orig_inj = biasconf.get_bias_value('IF_DC_P', default_biases)
print 'stopping the injection current...'
biasconf.set_bias(orig_inj[0],orig_inj[1],0)

#====================================stimulate the silicon
def random_stimulate(n_trials):

    for this_trial in range(n_trials):
        #pick a random neuron index
        neu_picked = np.random.randint(256)
        #print '############## picked neuron', neu_picked
        syn_neu_id = all_neu.synapses['virtual_exc'].addr['neu'] == neu_picked
        syn_virtual_id = all_neu.synapses['virtual_exc'].addr['syntype'] == 515 #stronger exc syn
        syn_virtual_this_neu = syn_neu_id & syn_virtual_id
        index_syn = np.where(syn_virtual_this_neu)[0]
        #print '############# SYN ADDR', all_neu.synapses['virtual_exc'][index_syn].addr
        #print '############# SYN ADDR', all_neu.synapses['virtual_exc'][index_syn].paddr
        synapse_virtual = all_neu.synapses['virtual_exc'][index_syn]
        spike_train = synapse_virtual.spiketrains_poisson(1000,duration=3000)
        setup.stimulate(spike_train,send_reset_event=False,duration=3000)
        id_neus_fired = mon_all.sl.raw_data()[:,1]
        index_fired = np.where(id_neus_fired == neu_picked)
        #print '############# fired neuron', np.unique(id_neus_fired)
        #print 'we selected neu: ', neu_picked
        #print 'we recorded neus: '
        print np.unique(id_neus_fired[index_fired])
        if( int(np.unique(id_neus_fired[index_fired])[0]) == neu_picked):
            print 'STIMULATION TEST num #'+str(this_trial)+' of '+str(n_trials)+' PASSED..'
        else:
            print 'STIMULATION TEST FAILED!!!!'
            print 'we are not able to stimulate the selected neuron via virtual synapses, please check addressed and biases'
            failed = True

print 'TEST single neuron stimulation via Virtual Synapses'
random_stimulate(3)

# one every ten neurons
print 'TEST stimulation of population neurons via Virtual Synapses'
synapse_virtual = all_neu[neu_to_stim].synapses['virtual_exc'][3::4]
spike_train = synapse_virtual.spiketrains_poisson(1000,duration=3000)
setup.stimulate(spike_train,send_reset_event=False,duration=3000)
pyNCS.monitors.RasterPlot([mon_stim, mon_back])
x_start = np.min(mon_all.sl.raw_data()[:,0])
x_stop = np.max(mon_all.sl.raw_data()[:,0])
xlim([x_start,x_stop])
tmp_rec = np.unique(mon_all.sl.raw_data()[:,1])
recorded = np.sort(tmp_rec)
stimulated = np.sort(neu_to_stim)
stimulated = np.append(stimulated,255) #we append our clock neuron

if( np.array_equal(recorded,stimulated) ):
    print 'POPULATION STIMULATION TEST PASSED...'
else:
    print 'POPULATION STIMULATION TEST FAILED !!!'
    failed = True



def read_synaptic_matrix(nsyn,freq=100,dur=500):
    '''
    read synaptic matrix
    '''
    matrixRead= np.zeros([256,256])
    outMon = pyNCS.monitors.SpikeMonitor(all_neu.soma)
    setup.monitors.import_monitors([outMon])
    for this_syn in range(nsyn):
        indx = all_neu[:].synapses['learning'].addr['syntype'] == this_syn
        spk = all_neu[:].synapses['learning'][indx].spiketrains_regular(freq,t_start=0,duration=dur)
        x = setup.stimulate(spk,send_reset_event=False,duration=dur)
        spikeout = outMon.sl
        rawOut = spikeout.raw_data()
        activeNeu = np.unique(rawOut[:,1])
        activeNeu = activeNeu.astype('int')
        #print activeNeu
        matrixRead[activeNeu,this_syn] = 1

    return matrixRead

print 'TEST plastic synapses pot/dep settings'
print "Program plastic weights random matrix ..."

matrix_set = np.zeros([256,256])#np.random.rand(256,256)
#matrix_set = np.ones([256,256])#np.random.rand(256,256)

colnum = 255
for i in range(colnum):
    matrix_set[:,i] = np.random.rand(1,256)

index_o = matrix_set<=0.5
index_z = matrix_set>=0.5
matrix_set[index_o] = 0
matrix_set[index_z] = 1

AERmn256r1.set_weight_matrix_plastic(matrix_set)

#print 'set weight plastic synapses to be very high...'
#orig_wplastic = biasconf.get_bias_value('PA_WHIDN_N', default_biases)
#biasconf.set_bias(orig_wplastic[0],orig_wplastic[1],90)

print 'now read synaptic matrix'
matrix_read = read_synaptic_matrix(255,freq=800,dur=100)
#imshow(matrix_read,interpolation='nearest')

#if( (matrix_read-matrix_set) != 0 ):
print "there were errors in settings the synapse"

index_row , index_col =np.where(matrix_read - matrix_set == 1)
synapses_wrong = len(np.where(index_row!=255)[0])
if(synapses_wrong > 0):
    print '############ we were not able to set n:'+str(synapses_wrong)+' plastic synapses'
else:
    print '############ all synapses were set correctly'

figure()
imshow(matrix_read-matrix_set,interpolation='nearest')
colorbar()

print 'setting back the injection current...'
biasconf.set_bias(orig_inj[0],orig_inj[1],int(orig_inj[2]))

if failed == True:
    print '#############################################################################'
    print '#############################################################################'
    print '################ TEST FAILED  !!!!!!!!!!!   #################################'
    print '################ PLEASE CHECK YOUR SETUP!!! #################################'
    print '#############################################################################'
else:
    print '#############################################################################'
    print '#############################################################################'
    print '################ ALL TESTS SUCCESFULLY PASSED ###############################'
    print '################ ENJOY YOUR SETUP !!!         ###############################'
    print '#############################################################################'
    

show()



