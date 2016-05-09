#!/usr/env/python

import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('../api/reservoir/')
sys.path.append('../api/retina/')
sys.path.append('../api/wij/')
sys.path.append('../api/rcn/')

#from wij import SynapsesLearning
from retina import Retina

#just do some recordings
record = 1
prefix='../'
setuptype = '../setupfiles/mc_final_mn256r1.xml'
setupfile = '../setupfiles/final_mn256r1_retina_monster.xml'
nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
chip = nsetup.chips['mn256r1']
nsetup.mapper._init_fpga_mapper()
p = pyNCS.Population('', '')
p.populate_all(nsetup, 'mn256r1', 'excitatory')

# we use the first 128 neurons to receive input from the retina
inputpop = pyNCS.Population('','')
inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  
#reset multiplexer
chip.configurator._set_multiplexer(0)
#init class synapses learning
#sl = SynapsesLearning(p, 'learning')
#matrix_learning = np.ones([256,256])

#init retina
ret = Retina(inputpop)
#program all excitatory synapses for the programmable syn
matrix_exc = np.ones([256,256])
nsetup.mapper._program_onchip_exc_inh(matrix_exc)
#set to zeros recurrent and learning synapses
matrix_off = np.zeros([256,256])
matrix_rec_off = np.zeros([256,512])
nsetup.mapper._program_onchip_programmable_connections(matrix_off)
nsetup.mapper._program_onchip_recurrent(matrix_rec_off)
matrix_weights = np.ones([256,256])
nsetup.mapper._program_onchip_weight_matrix_programmable(matrix_weights)

#ret.map_retina_to_mn256r1()
#ret.map_retina_to_mn256r1(inputpop) #this function map retina output pixels to mn256r1 programmable syn inputpop.synapses['programmable'].paddr[1::2]
#we first init the mappings
ret._init_fpga_mapper()
retina, pre_address, post_address  = ret.map_retina_to_mn256r1_macro_pixels()

#exc inh 
all_exc= np.ones([256,256])
nsetup.mapper._program_onchip_exc_inh(all_exc)
chip.configurator._set_all_neu_tau2()

ion()
def go_reconstruct():
    monitor = pyNCS.monitors.SpikeMonitor(ret.pop_dest.soma)
    nsetup.monitors.import_monitors([monitor])
    nsetup.stimulate({},duration=1000, send_reset_event=False)
    stim = ret.reconstruct_stim(monitor,bb=2)
    for i in range(len(stim)):
        figure()
        imshow(stim[i],interpolation="nearest")
        colorbar()

#rcn connections
#make rcn connection with the rest of programmable synapses
#rcn = Rcn(inputpop,cee=0.1)
#nsetup.mapper._program_onchip_weight_matrix_programmable(rcn.matrix_programmable_w)
#nsetup.mapper._program_onchip_programmable_connections(rcn.matrix_programmable_rec)
#nsetup.mapper._program_onchip_programmable_connections(rcn.matrix_programmable_exc_inh)

#import monitor for chip
#rcnmon =  pyNCS.monitors.SpikeMonitor(rcn.mypop_e.soma)
#nsetup.monitors.import_monitors([rcnmon])
#record for some time
#nsetup.stimulate({},duration=2000)
#stim = rcn.reconstruct_stim(rcnmon,bb=2)
#imshow(np.rot90(stim[1]), interpolation="nearest")

#kmeans clustering
#from scipy.cluster.vq import vq, kmeans, whiten
#centroids, distortion = kmeans(data, 10)
#time_rec = [np.min(rcn_raw_data[:,0]),np.max(rcn_raw_data[:,0])] 
#c_time = np.linspace(0, time_rec[1]-time_rec[0], bb) 
#max_prob_class = np.argmax(centroids, axis=0)
#figure()
#plot(c_time,max_prob_class,'o')
#figure()
#imshow(data)
#show()
