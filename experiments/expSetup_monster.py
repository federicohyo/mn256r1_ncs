#!/usr/env/python

import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('../api/retina/')
sys.path.append('../api/wij/')
from wij import SynapsesLearning

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

#init class synapses learning
sl = SynapsesLearning(p, 'learning')

