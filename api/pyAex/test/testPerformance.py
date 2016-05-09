import pyNCS
import pyAex
import os
import pyST
import threading
import time
import numpy as np
#import pyNCS.AerViewer





setup=pyNCS.Setup()
setup.load('zenzero.xml')
setup.apply()

client=pyAex.netMonStimEventsQueue(host='localhost',autostart=0)

class virtualEventsWriter ( threading.Thread ):
    def __init__(self,aer_events,fps):
        self.device_fd=os.open('/tmp/aerfx2_virtual',os.O_RDWR)
        self.aer_data=np.loadtxt(aer_events,'uint32')        #Consider reading line by line for very large files
        self.fTus=np.uint32(1e6/fps)
        threading.Thread.__init__ ( self )
        self.daemon=True

    def put( self, t0, t ):
        '''
        t0 start time (us)
        t end time (us)
        '''
        ev_intval=self.aer_data[(self.aer_data[:,0]>t0)*(self.aer_data[:,0]<t)]
        os.write(self.device_fd,ev_intval)
    def run ( self ):
        t0=time.time()
        while True:
            t0_fus=np.uint32((time.time()-t0)*1e6)
            self.put(t0_fus,t0_fus+self.fTus)
            t_fus=np.uint32((time.time()-t0)*1e6)
            time.sleep(self.fTus*1e-6-(t_fus-t0_fus)*1e-6)

v=virtualEventsWriter('data.txt',40)

v.start()
client.run()

    #viewer=pyNCS.AerViewer.Aer2DViewer(0,'zenzero')    
    



