from pyST import *
import pyNCS
import pyAex
import os
import pyST
import threading
import time
import sys
import numpy as np
from pyAexServer import ServerStarter


setup=pyNCS.Setup(filename='setups/'+sys.argv[1]+'_setuptype.xml')
setup.load('setups/'+sys.argv[1]+'.xml')
setup.apply()
#
if sys.argv[1]=='localhost':
    aexs=ServerStarter.start_server()
client=pyAex.netClient(host=sys.argv[1],autostart=1,fps=1)
#vSeq=pyAex.virtEventSeq('data.txt',25)
#vSeq.start()
b=SpikeList([],[])
c=SpikeList([],[])
for i in range(0,120):
    b[i+.0625]=STCreate.poisson_generator(500000, t_stop=100)
client.stimulate({3:b},isi=False, verbose=True)

client.stop()
ServerStarter.stop_server(aexs)








    



