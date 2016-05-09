import pyNCS
import pyAex
import pyST
import time
import numpy as np
import unittest
from pyAexServer import ServerStarter

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.setup=pyNCS.Setup('setups/zenzero_setuptype.xml')
        self.setup.load('setups/localhost.xml')
        self.setup.apply()
        self.aexs=ServerStarter.start_server()
        self.client=pyAex.netClient(host='localhost',autostart=1)
        self.vSeq=pyAex.virtEventSeq(self.aexs.aexDevice, 'shortdata.txt',25)

    def testStimulation(self):
        self.client.flush()
        b=pyST.SpikeList([],[])
        b[114.0625]=pyST.STCreate.poisson_generator(500)
        b[4.0625]=pyST.STCreate.poisson_generator(500)
        o=self.client.stimulate({5:b},isi=False)[5]
        b.time_offset(offset=-b.raw_data()[0,0])
        brd=b.raw_data()
        ord=o.raw_data()
        for i in xrange(len(b.raw_data())):
            self.assertAlmostEqual(ord[i][0],brd[i][0],2)
            self.assertAlmostEqual(ord[i][1],brd[i][1],2)


    def testEventConservation(self):
        "Tests functionality of client.listen, client.fetch and normalizeAER"
        self.client.flush()
        self.vSeq.run()
        listen_out=self.client.listen(verbose=False,output='spikelist')
        res=pyST.getDefaultMonChannelAddress().exportAER(listen_out,format='t',isi=False)
        stim=pyST.events(np.fliplr(self.vSeq.aer_data))
        nEvents_listen=len(stim)
        nEvents_aerdat=len(res)

        #Is the number of events conserved?
        self.assert_( nEvents_listen == nEvents_aerdat )
       
        #Are the addresses conserved?
        for i in stim.get_ad():
            self.assert_( i in res.get_ad())

    def tearDown(self):
        self.client.stop()
        self.aexs=ServerStarter.stop_server()
        
if __name__ == '__main__':
    #TestSequenceFunctions('testStimulation').debug()
    unittest.main()

