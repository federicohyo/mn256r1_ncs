# mapper gui
# author: Marc Osswald - marc@ini.phys.ethz.ch
# Mar 2014

from traits.api import HasTraits, Str, Int, Range, Array, List, File, Float, Enum, Instance, Bool
from traitsui.api import View, Item, Group, RangeEditor, Spring
import traitsui
from traits.api import Button as traitsButton

# Import libraries
from pylab import *
import numpy as np
from PCIeMapperConfigurator import PCIeMapperConfigurator
from BasicMapper import BasicMapper
from multiprocessing import Process, Lock

class mapperProcess(HasTraits):

    #fields
    pcieconf = PCIeMapperConfigurator
    mapper = BasicMapper
    p = Process

    #traits
    mapperType = Enum('BasicMapper')
    startMapper = traitsButton()
    stopMapper = traitsButton()

    view = View(Group(Group(Item(name='mapperType', label='Select instance'),
                            Spring(springy=True),
                            Item(name='startMapper', label='Run', show_label=False, enabled_when='isNotRunning'),
                            Item(name='stopMapper', label='Stop', show_label=False, enabled_when='isRunning'),
                            layout='normal', style='simple', orientation='horizontal'),
                      orientation='vertical'))

    def run(self):
        self.lock.acquire()
        #do mapping
        while True:
            addr = self.pcieconf.get_32()
            print(addr)
            #self.pcieconf.send_32(addr)

    def terminateMapperProcess(self):
        if self.isRunning:
            self.isRunning = False
            self.isNotRunning = True
            self.p.terminate()
            self.lock.release()

    def __init__(self, pcieconf, lock):
        self.lock = lock
        self.pcieconf = pcieconf
        self.isRunning = False
        self.isNotRunning = True

    def _startMapper_fired(self):
        if self.isNotRunning:
            #instantiate mapper
            self.mapper = BasicMapper()
            #start mapper process
            self.p = Process(target=self.run)
            self.p.start()
            self.isRunning = True
            self.isNotRunning = False

    def _stopMapper_fired(self):
        if self.isRunning:
            self.terminateMapperProcess()
            del self.mapper


class mapperTraits(HasTraits):

    #pci configurator
    SEQ_FPGA_ID = Range(0,3,1, mode='spinner')
    SEQ_FPGA_CHNL = Range(0,0,0, mode='spinner')
    MON_FPGA_ID = Range(0,3,0, mode='spinner')
    MON_FPGA_CHNL = Range(0,0,0, mode='spinner')
    pcieconf = Instance(PCIeMapperConfigurator)
    close_pcie_seq = traitsButton()
    close_pcie_mon = traitsButton()
    status_pcie_seq = Str('undefined')
    status_pcie_mon = Str('undefined')
    pcie_flag_seq = False
    pcie_flag_not_seq = True
    pcie_flag_mon = False
    pcie_flag_not_mon = True

    #control buttons
    open_devices = traitsButton()
    interface = Enum('PCIe')

    #stats
    in_rate = Str('0000.0 kEvt/s')
    out_rate = Str('0000.0 kEvt/s')

    #mapper process
    mapper = Instance(mapperProcess(PCIeMapperConfigurator, Lock))


    traits_view = View(Group(Group(Item(name='in_rate', label='Input rate', style='readonly', emphasized=True),
                                   Spring(springy=True),
                                   Item(name='out_rate', label='Output rate', style='readonly', emphasized=True),
                                   label='Statistics', orientation='horizontal', show_border=True),
                             Group(Item(name='mapper', style='custom', show_label=False),
                                   label='Configuration', show_border=True),
                             label='Mapper'),
                       Group(Group(Spring(springy=True),
                                   Item(name='interface', label='Please choose an interface'),
                                   Item('open_devices', label='Open Devices', show_label=False),
                                   label='Interface', show_border=True, orientation='horizontal'),
                             Group(Item(name='MON_FPGA_ID', label='Device ID', enabled_when='pcie_flag_not_mon'),
                                   Item(name='MON_FPGA_CHNL', label='Channel', enabled_when='pcie_flag_not_mon'),
                                   Spring(springy=True),
                                   Item(name='status_pcie_mon', label='Status', style='readonly', emphasized=True),
                                   Item(name='close_pcie_mon', label='Close', show_label=False, enabled_when='pcie_flag_mon'),
                                   label='PCIe Monitor', show_border=True, orientation='horizontal'),
                             Group(Item(name='SEQ_FPGA_ID', label='Device ID', enabled_when='pcie_flag_not_seq'),
                                   Item(name='SEQ_FPGA_CHNL', label='Channel', enabled_when='pcie_flag_not_seq'),
                                   Spring(springy=True),
                                   Item(name='status_pcie_seq', label='Status', style='readonly', emphasized=True),
                                   Item(name='close_pcie_seq', label='Close', show_label=False, enabled_when='pcie_flag_seq'),
                                   label='PCIe Sequencer', show_border=True, orientation='horizontal'),
                             label='Setup'))

    def _anytrait_changed(self):
        if not(self.mapper is None):
            self.mapper.terminateMapperProcess()

    def openDefaultDevices(self):
        self.lock.acquire()
        self.pcieconf.openPCIeMon(self.MON_FPGA_ID,self.MON_FPGA_CHNL)
        self.pcieconf.openPCIeSeq(self.SEQ_FPGA_ID,self.SEQ_FPGA_CHNL)
        self.status_pcie_mon = 'opened'
        self.status_pcie_seq = 'opened'
        self.pcie_flag_mon = True
        self.pcie_flag_not_mon = False
        self.pcie_flag_seq = True
        self.pcie_flag_not_seq = False
        self.lock.release()

    def _open_devices_fired(self):
        self.lock.acquire()
        if self.interface == 'PCIe':
            if self.pcie_flag_seq == False:
                self.pcieconf.openPCIeSeq(self.SEQ_FPGA_ID,self.SEQ_FPGA_CHNL)
                self.status_pcie_seq = 'opened'
                self.pcie_flag_seq = True
                self.pcie_flag_not_seq = False
            if self.pcie_flag_mon == False:
                self.pcieconf.openPCIeMon(self.MON_FPGA_ID,self.MON_FPGA_CHNL)
                self.status_pcie_mon = 'opened'
                self.pcie_flag_mon = True
                self.pcie_flag_not_mon = False
        self.lock.release()

    def _close_pcie_seq_fired(self):
        self.lock.acquire()
        if self.pcie_flag_seq == True:
            self.pcieconf.closePCIeSeq()
            self.status_pcie_seq = 'closed'
            self.pcie_flag_seq = False
            self.pcie_flag_not_seq = True
        self.lock.release()

    def _close_pcie_mon_fired(self):
        self.lock.acquire()
        if self.pcie_flag_mon == True:
            self.pcieconf.closePCIeMon()
            self.status_pcie_mon = 'closed'
            self.pcie_flag_mon = False
            self.pcie_flag_not_mon = True
        self.lock.release()


#dvs configurator object
configurator = PCIeMapperConfigurator()

#lock
lock = Lock()

#init values
def_mapper = mapperTraits(mapper=mapperProcess(configurator, lock), lock=lock, pcieconf=configurator, name='mapper')

def_mapper.openDefaultDevices()

def_mapper.configure_traits()
