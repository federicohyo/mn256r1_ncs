__author__ = 'marc'

from traits.api import HasTraits, Str, Int, Range, Array, List, File, Float, Enum, Instance
from traitsui.api import View, Item, Group, RangeEditor, Spring
import traitsui
from traits.api import Button as traitsButton
from MN256R1Configurator import MN256R1Configurator
import time

# Import libraries
from pylab import *
import numpy as np

class AerConfiguratorGroup(HasTraits):

    dpiNames = {'NPDPIE': 0,
                'NPDPII': 1,
                'PDPIE': 2,
                'VDPIE': 3,
                'VDPII': 4
    }

    plasticLatches = {'Recurrent': 0,
                      'Broadcast': 1
    }

    non_plasticLatches = {'Recurrent': 0,
                      'Exc_Inh': 1,
                      'W0': 2,
                      'W1': 3,
                      'Broadcast': 4
    }


    configurator = MN256R1Configurator
    mon_neuron = Range(0,255)
    mon_dpi_neuron = Range(0,255)
    set_monitor = traitsButton()
    set_dpi_monitor = traitsButton()
    dpi = Enum('NPDPIE','NPDPII','PDPIE','VDPIE','VDPII')
    reset_mon = traitsButton()
    #neuron
    set_neuron = Range(0,255)
    set_neuron_latch = traitsButton()
    reset_neuron_latch = traitsButton()
    enable_reset_buffer = traitsButton()
    disable_reset_buffer = traitsButton()
    #plastic synapses
    set_plastic_latch = traitsButton()
    reset_plastic_latch = traitsButton()
    set_plastic_weight_hi = traitsButton()
    set_plastic_weight_lo = traitsButton()
    plastic_latch = Enum('Recurrent', 'Broadcast')
    plastic_row = Range(0,255)
    plastic_col = Range(0,255)
    #non_plastic synapses
    set_non_plastic_latch = traitsButton()
    reset_non_plastic_latch = traitsButton()
    non_plastic_latch = Enum('Recurrent', 'Exc_Inh', 'W0', 'W1', 'Broadcast')
    non_plastic_row = Range(0,255)
    non_plastic_col = Range(0,255)

    #Multiplexer
    mux_value = Range(0,8)
    set_mux = Range(0,8)
 
    set_mux = traitsButton()

    #Send Test Spikes
    row = Range(0,255)
    col = Range(0,519)
    col_bc = Range(0,511)
    set_plastic_monitor = traitsButton()
    spike_frequency = Range(1,500)
    spike_count = Range(0,10000)
    start_test_sequencer = traitsButton()
    spike_frequency_bc = Range(1,500)
    spike_count_bc = Range(0,10000)
    start_test_sequencer_bc = traitsButton()

    view = View(Group(Group(Group(Spring(springy=True),
                                  Item(name='mon_neuron',label='Membrane potential of neuron'),
                                  Item(name='set_monitor',label='Set Monitor', show_label=False),
                                  orientation='horizontal'),
                            Group(Spring(springy=True),
                                  Item(name='dpi', label='DPI current of'),
                                  Item(name='mon_dpi_neuron',label='of neuron'),
                                  Item(name='set_dpi_monitor',label='Set Monitor', show_label=False),
                                  orientation='horizontal'),
                            Group(Spring(springy=True),
                                  Item(name='reset_mon',label='Resset All Monitors', show_label=False),
                                  orientation='horizontal'),
                            show_border=True,
                            label='Monitor'),
###
                      Group(Group(Item(name='set_neuron',label='Number of neuron'),
                                  Spring(springy=True),
                                  Item(name='set_neuron_latch', label='Set Bias1', show_label=False),
                                  Item(name='reset_neuron_latch', label='ReSet All to Bias2', show_label=False),
                                  orientation='horizontal'),
                            Group(Spring(springy=True),
                                  Item(name='enable_reset_buffer', label='Enable Reset Buffer of Neuron', show_label=False),
                                  Item(name='disable_reset_buffer', label='Disable Reset Buffer of Neuron', show_label=False),
                                  orientation='horizontal'),
                            show_border=True,
                            label='Neuron'),
###
                      Group(Group(Item(name='plastic_col',label='Column'),
                                  Item(name='plastic_row',label='Row'),
                                  Spring(springy=True),
                                  Item(name='set_plastic_monitor',label='Set Monitor',show_label=False),
                                  orientation='horizontal'),
                            Group(Spring(springy=True),
                                  Item(name='plastic_latch', label='Configure latches'),
                                  Item(name='set_plastic_latch', label='Set', show_label=False),
                                  Item(name='reset_plastic_latch', label='Reset', show_label=False),
                                  orientation='horizontal'),
                            Group(Spring(springy=True),
                                  Item(name='set_plastic_weight_hi', label='Set Weight High', show_label=False),
                                  Item(name='set_plastic_weight_lo', label='Set Weight Low', show_label=False),
                                  orientation='horizontal'),
                            show_border=True,
                            label='Plastic Synapses'),
###
                     Group(Group(Item(name='non_plastic_col',label='Column'),
                                  Item(name='non_plastic_row',label='Row'),
                                  Spring(springy=True),
                                  Item(name='non_plastic_latch', label='Configure latches'),
                                  Item(name='set_non_plastic_latch', label='Set', show_label=False),
                                  Item(name='reset_non_plastic_latch', label='Reset', show_label=False),
                                  orientation='horizontal'),
                            show_border=True,
                            label='Non Plastic Synapses'),
### Multiplexer
                     Group(Group(Spring(springy=True),
                                 Item(name='mux_value',label='Mux'),
                                 Item(name='set_mux',label='Set Mux', show_label=False),
                                 orientation='horizontal'),
                            show_border=True,
                            label='Multiplexer'),
###


                      Group(Group(Item(name='col',label='Column'),
                                  Item(name='row',label='Row'),
                                  Spring(springy=True),
                                  Item(name='spike_count',label='Number of spikes'),
                                  Item(name='spike_frequency',label='Frequency (Hz)'),
                                  Item(name='start_test_sequencer',label='Send',show_label=False),
                                  orientation='horizontal'),
                            Group(Item(name='col_bc',label='Broadcast Column'),
                                  Spring(springy=True),
                                  Item(name='spike_count_bc',label='Number of spikes'),
                                  Item(name='spike_frequency_bc',label='Frequency (Hz)'),
                                  Item(name='start_test_sequencer_bc',label='Send',show_label=False),
                                  orientation='horizontal'),
                            show_border=True,
                            label='Send Test Spikes'),
                      orientation='vertical'))

    def __init__(self, conf):
        self.configurator = conf

    def _start_test_sequencer_fired(self):
        for i in range(0,self.spike_count):
            self.configurator.set_aer(self.col*256+self.row)
            time.sleep(1/float(self.spike_frequency))
    def _start_test_sequencer_bc_fired(self):
        for i in range(0,self.spike_count_bc):
            self.configurator.set_aer(self.col_bc+133120)
            time.sleep(1/float(self.spike_frequency))


    def _set_plastic_weight_hi_fired(self):
        self.configurator.set_aer(1*256*256+self.plastic_col*256+self.plastic_row+1051136)
    def _set_plastic_weight_lo_fired(self):
        self.configurator.set_aer(self.plastic_col*256+self.plastic_row+1051136)


    def _set_neuron_latch_fired(self):
        self.configurator.set_aer(self.set_neuron+1249026)
        self.configurator.set_aer(1249283)                     #PLEASE FIX!
        self.configurator.set_aer(self.set_neuron+1249026)

    def _reset_neuron_latch_fired(self):
        self.configurator.set_aer(1249282)
        self.configurator.set_aer(1249283)

    def _enable_reset_buffer_fired(self):
        self.configurator.set_aer(1249540)
    def _disable_reset_buffer_fired(self):
        self.configurator.set_aer(1249541)
        self.configurator.set_aer(self.mon_neuron+1249284)     #PLEASE FIX!
        self.configurator.set_aer(1249541)
        self.configurator.set_aer(self.mon_neuron+1249284)     #PLEASE FIX!
    def _set_plastic_latch_fired(self):
        self.configurator.set_aer(self.plasticLatches[self.plastic_latch]*2*256*256+1*256*256+self.plastic_col*256+self.plastic_row+788992)

    def _reset_plastic_latch_fired(self):
        self.configurator.set_aer(self.plasticLatches[self.plastic_latch]*2*256*256+0*256*256+self.plastic_col*256+self.plastic_row+788992)

    def _set_non_plastic_latch_fired(self):
        self.configurator.set_aer(self.non_plasticLatches[self.non_plastic_latch]*2*256*256+1*256*256+self.non_plastic_col*256+self.non_plastic_row+133632)

    def _reset_non_plastic_latch_fired(self):
        self.configurator.set_aer(self.non_plasticLatches[self.non_plastic_latch]*2*256*256+0*256*256+self.non_plastic_col*256+self.non_plastic_row+133632)


    def _set_plastic_monitor_fired(self):
        self.setMonitorNeuron()
        self.configurator.set_aer(self.plastic_col*256+self.plastic_row+1182208)

    def _reset_mon_fired(self):
        self.configurator.set_aer(1249024)
 
    def setMultiplexer(self):
        self.configurator.set_aer(self.mux_value+1249542)

    def _set_mux_fired(self):
        self.setMultiplexer()

    def _mux_value_changed(self):
        self.setMultiplexer()



    def setMonitorNeuron(self):
        self.configurator.set_aer(1249024)
        self.configurator.set_aer(1249025)
        self.configurator.set_aer(self.mon_neuron+1249284) #PLEASE FIX!
        self.configurator.set_aer(1249025)
        self.configurator.set_aer(self.mon_neuron+1249284)

    def _set_monitor_fired(self):
        self.setMonitorNeuron()

    def _mon_neuron_changed(self):
        self.setMonitorNeuron()

    def setMonitorDPI(self):
        self.configurator.set_aer(self.dpiNames[self.dpi]*256+self.mon_dpi_neuron+1247744)

    def _set_dpi_monitor_fired(self):
        self.setMonitorDPI()

    def _mon_dpi_neuron_changed(self):
        self.setMonitorDPI()

    def _dpi_changed(self):
        self.setMonitorDPI()
