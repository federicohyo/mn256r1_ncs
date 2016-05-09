#!/usr/env/python

import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('../api/nef/')

#parameters and actions 
n_steps = 10
do_figs_encoding = False

#populations divisible by 2 for encoders
neuron_ids = [np.linspace(0,127,128),np.linspace(128,255,128)]
npops = len(neuron_ids)

#setup
prefix='../'
setuptype = '../setupfiles/mc_final_mn256r1.xml'
setupfile = '../setupfiles/final_mn256r1_retina_monster.xml'
nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
chip = nsetup.chips['mn256r1']
#nsetup.mapper._init_fpga_mapper()
chip.configurator._set_multiplexer(0)

#populate neurons
nefpops = [pyNCS.Population('neurons', 'for fun') for i in range(len(neuron_ids))]
for this_pop in range(npops):
    nefpops[this_pop].populate_by_id(nsetup,'mn256r1','excitatory', neuron_ids[this_pop])
#populate broadcast pop
b_nefpop = pyNCS.Population('neurons', 'for fun') 
b_nefpop.populate_by_id(nsetup,'mn256r1','excitatory', np.linspace(0,255,256)) #good luck in understanding this...

#encoders
encoders = [np.zeros(len(neuron_ids[i])) for i in range(npops)]
for this_pop in range(npops):
    encoders[this_pop] = np.concatenate([np.ones(len(neuron_ids[this_pop])/2)*-1,np.ones(len(neuron_ids[this_pop])/2)]) 

#init nef on neuromorphic chips
import nef
ion()
input_space = np.linspace(-1,1,n_steps)
myNef = nef.Nef(nefpops,b_nefpop,encoders)
myNef._init_fpga_mapper()
myNef.program_encoders()                # program encoders by using different taus
myNef.generate_biases()                 # generate synaptic biases for neurons
myNef.program_input_weights(nsyn = 256, w = 1) # program input weight using biases
myNef.measure_synaptic_efficacy_prob()
## 0
#tuning curves
myNef.measure_tuning_curves(min_freq = 0, max_freq=500, nsteps=n_steps, nsyn=256)
myNef.plot_tuning_curves()
myNef.plot_syn_eff_prob()
## 1
#encode a function
myNef.encode_function(nsteps=n_steps,exponent=2)
myNef.plot_encoded_function(nsteps=n_steps,input_space=input_space)
if(do_figs_encoding):
    #try to different stimulation to see how noise effect us
    for i in range(3):
        myNef.measure_tuning_curves(min_freq = 0, max_freq=500, nsteps=n_steps, nsyn=256)
        myNef.encode_function(find_decoders=False,nsteps=n_steps,exponent=2)
        myNef.plot_encoded_function(nsteps=n_steps,input_space=input_space)
## 2
#transform and communication channels
#calculate weights
decoders_a = myNef.compute_decoders(myNef.pops[0], nsteps=10, exponent=1, method='pinv') # nsteps depends on the measured tuning curves
decoders_b = myNef.compute_decoders(myNef.pops[1], nsteps=10, exponent=1, method='pinv')
decoders_a = np.reshape(decoders_a,[len(decoders_a),1])
weights = np.dot(decoders_a,[myNef.encoders[1]])
#program weights
#myNef.program_weights_prob(myNef.pops[0], myNef.pops[1], weights)
#enable detailed mapping for interface
#nsetup.mapper._program_detail_mapping(2**0) #chip interface

