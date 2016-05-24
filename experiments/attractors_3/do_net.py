# === Include libraries =======================================================

import numpy as np
import conf_mn256r1
import matplotlib
from pylab import *

# === Parameters determining model dynamics ===================================

# Network structure
p       = 3	    #pop number (exc and inh)
Ne      = 65	#exc neurons
Ni      = 10	#inh neurons
Ne_tot  = p*Ne	#
Ni_tot  = p*Ni	#

Cee 	= 0.7 #0.7
Cei	    = 0.4
Cie	    = 0.15
Cii	    = 0.4 #self-inhibition, de-synch

Cdiag_E = 0.0 #0.4
Cdiag_I = 0.1 #0.4

# === Neural populations ======================================================

print "Setting up the populations ..."

# Excitatory subpopulations:
popsne = [conf_mn256r1.populate_neurons(Ne, order=True) for i in range(p)] 
popsni = [conf_mn256r1.populate_neurons(Ni, order=True) for i in range(p)] 

#choose pop neurons 
matrix_learning = np.zeros([256,256])
matrix_pot = np.zeros([256,256])
for i in range(p):
#    conf_mn256r1.connect_populations_programmable(popsne[i],popsne[i],Cee,[2,2])
    conf_mn256r1.connect_populations_learning(popsne[i],popsne[i],Cee,1)
    conf_mn256r1.connect_populations_programmable(popsne[i],popsni[i],Cei,[0,3])
    conf_mn256r1.connect_populations_programmable_inh(popsni[i],popsne[i],Cie,[1,1])
    conf_mn256r1.connect_populations_learning(popsni[i],popsni[i],Cii,[1,1])
#    conf_mn256r1.connect_populations_programmable(popsni[i],popsni[i],Cii,[2,2])

for i in range(p):
    for j in range(p):
#        if(i != j):
#            w_conn_exc = 1#np.random.randint(2)+1
            w_conn_inh = 2#np.random.randint(2)+1
#            conf_mn256r1.connect_populations_programmable(popsne[i],popsne[j],Cdiag_E,[w_conn_exc,w_conn_exc])
            conf_mn256r1.connect_populations_programmable_inh(popsne[i],popsne[j],Cdiag_I,[w_conn_inh,w_conn_inh])

# === end configuration ======================================================


