## new way of reading #Imports for AER monitoring
import pyAex
from pyNCS.pyST import *
import pyNCS, pyNCS.pyST.STas

class Stimulator:
    def __init__(self, gui, host='localhost', port=50001, channel=0, dims=(128, 0), fps=25):
        self.gui = gui
        self.port = port
        self.host = host
        self.dims = dims
        self.fps = fps
        self.channel = channel
        self.tDuration = 5.
        self.gui.updater = self
        self.stcs = getDefaultMonChannelAddress()
        pyNCS.pyST.STas.addrBuildHashTable(self.stcs[channel])
        self.eventsQueue = pyAex.aexclient.AEXMonClient(MonChannelAddress=self.stcs,
                                                   channels=[
                                                       self.channel],
                                                   host=self.host,
                                                   port=self.port,
                                                   autostart=True,
                                                   fps=self.fps)
        self.neurons = numpy.array([])
        self.times = numpy.array([])

        self.tot_neurons = numpy.array([])
        self.tot_times = numpy.array([])
        #self.z = numpy.zeros(self.dims) #Data being plotted

        self.gui.channel = channel

        #count for updating the rate
        self.clockcount = 0
        self.ratecount = 0


