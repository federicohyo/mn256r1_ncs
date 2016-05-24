# mn256 biases gui
# March 2014
# authors marc@ini.phys.ethz.ch, federico@ini.phys.ethz.ch

from traits.api import HasTraits, Str, Int, Range, Array, List, File, Float, Enum, Instance
from traitsui.api import View, Item, Group, RangeEditor, Spring
import traitsui
from traits.api import Button as traitsButton

# Import libraries
from pylab import *
import numpy as np
from MN256R1Configurator import MN256R1Configurator
from AerConfiguratorGroup import AerConfiguratorGroup
from PathConfiguratorGroup import PathConfiguratorGroup

import time

import sys
sys.path.append('biasgenlib/')

class biasGroup(HasTraits):
    name = str
    coarse = Enum('24u', '3.2u', '0.4u', '50n', '6.5n', '820p', '105p', '15p')
    fine = Range(0, 255)
    current = Str
    lowHigh = Enum('HighBias', 'LowBias')
    type = Enum('NBias', 'PBias')
    default = traitsButton()
    view = View(Group(Item(name='name', style='readonly', show_label=False, emphasized=True),
                      Spring(springy=True),
                      Item(name='coarse', show_label=False),
                      Item(name='fine', show_label=False),
                      Item(name='current'),
                      Item(name='lowHigh', show_label=False),
                      Item(name='type', show_label=False),
                      Item(name='default', label='Default', show_label=False),
                      orientation='horizontal',
                      layout='normal',
                      style='simple'))

    def __init__(self, n):
        self.name = n

    def sentBias(self):
        configurator.set_bias(self.name, self.coarse, self.fine, biasLowHigh=self.lowHigh, biasType=self.type,
                          biasCascode='', biasEnable='')

    def exportToString(self):
        return self.name+','+self.coarse+','+str(self.fine)+','+self.lowHigh+',Normal,'+self.type+',BiasEnable,'

    def updateCurrent(self):
        unit = self.coarse[-1]
        maxCurrent = float(self.coarse.split(unit)[0])
        current = self.fine * maxCurrent / 256
        self.current = str(round(current, 4)) + unit
        self.sentBias()

    def _fine_changed(self):
        self.updateCurrent()

    def _coarse_changed(self):
        self.updateCurrent()

    def _lowHigh_changed(self):
        self.updateCurrent()

    def _type_changed(self):
        self.updateCurrent()

    def _default_fired(self):
        self.setDefaults('')

    def setDefaults(self, file):
        if file == '':
            defaults = configurator.read_default_bias(self.name)
        else:
            defaults = configurator.read_default_bias(self.name, def_file=file)
        self.coarse = defaults[1]
        self.fine = int(defaults[2])
        self.lowHigh = defaults[3]
        self.type = defaults[5]
        self.updateCurrent()

class biasBufferGroup(HasTraits):
    name = str
    coarse = Enum('24u', '3.2u', '0.4u', '50n', '6.5n', '820p', '105p', '15p')
    fine = Range(0, 255)
    current = Str
    default = traitsButton()
    view = View(Group(Item(name='name', style='readonly', show_label=False, emphasized=True),
                      Spring(springy=True),
                      Item(name='coarse', show_label=False),
                      Item(name='fine', show_label=False),
                      Item(name='current'),
                      Item(name='default', label='Default', show_label=False),
                      orientation='horizontal',
                      layout='normal',
                      style='simple'))

    def __init__(self, n):
        self.name = n

    def sentBias(self):
        configurator.set_bias(self.name, self.coarse, self.fine, biasLowHigh='NONE', biasType='NONE',
                          biasCascode='NONE', biasEnable='NONE')

    def exportToString(self):
        return self.name+','+self.coarse+','+str(self.fine)+','+'NONE,NONE,NONE,NONE,'

    def updateCurrent(self):
        unit = self.coarse[-1]
        maxCurrent = float(self.coarse.split(unit)[0])
        current = self.fine * maxCurrent / 256
        self.current = str(round(current, 4)) + unit
        self.sentBias()

    def _fine_changed(self):
        self.updateCurrent()

    def _coarse_changed(self):
        self.updateCurrent()

    def _default_fired(self):
        self.setDefaults('')

    def setDefaults(self, file):
        if file == '':
            defaults = configurator.read_default_bias(self.name)
        else:
            defaults = configurator.read_default_bias(self.name, def_file=file)
        self.coarse = defaults[1]
        self.fine = int(defaults[2])
        self.updateCurrent()

class biasSpecialGroup(HasTraits):
    name = str
    value = Range(0, 63)
    default = traitsButton()
    view = View(Group(Item(name='name', style='readonly', show_label=False, emphasized=True),
                      Spring(springy=True),
                      Item(name='value', show_label=False),
                      Item(name='default', label='Default', show_label=False),
                      orientation='horizontal',
                      layout='normal',
                      style='simple'))

    def __init__(self, n):
        self.name = n

    def sentBias(self):
        configurator.set_bias(self.name, 'SPECIAL', self.value, biasLowHigh='NONE', biasType='NONE',
                          biasCascode='NONE', biasEnable='NONE')

    def exportToString(self):
        return self.name+',SPECIAL,'+str(self.value)+','+'NONE,NONE,NONE,NONE,'

    def _value_changed(self):
        self.sentBias()

    def _default_fired(self):
        self.setDefaults('')

    def setDefaults(self, file):
        if file == '':
            defaults = configurator.read_default_bias(self.name)
        else:
            defaults = configurator.read_default_bias(self.name, def_file=file)
        self.value = int(defaults[2])


class biasTraits(HasTraits):

    #biases
    bias0 = Instance(biasGroup(str))
    bias1 = Instance(biasGroup(str))
    bias2 = Instance(biasGroup(str))
    bias3 = Instance(biasGroup(str))
    bias4 = Instance(biasGroup(str))
    bias5 = Instance(biasGroup(str))
    bias6 = Instance(biasGroup(str))
    bias7 = Instance(biasGroup(str))
    bias8 = Instance(biasGroup(str))
    bias9 = Instance(biasGroup(str))
    bias10 = Instance(biasGroup(str))
    bias11 = Instance(biasGroup(str))
    bias12 = Instance(biasGroup(str))
    bias13 = Instance(biasGroup(str))
    bias14 = Instance(biasGroup(str))
    bias15 = Instance(biasGroup(str))
    bias16 = Instance(biasGroup(str))
    bias17 = Instance(biasGroup(str))
    bias18 = Instance(biasGroup(str))
    bias19 = Instance(biasGroup(str))
    bias20 = Instance(biasGroup(str))
    bias21 = Instance(biasGroup(str))
    bias22 = Instance(biasGroup(str))
    bias23 = Instance(biasGroup(str))
    bias24 = Instance(biasGroup(str))
    bias25 = Instance(biasGroup(str))
    bias26 = Instance(biasGroup(str))
    bias27 = Instance(biasGroup(str))
    bias28 = Instance(biasGroup(str))
    bias29 = Instance(biasGroup(str))
    bias30 = Instance(biasGroup(str))
    bias31 = Instance(biasGroup(str))
    bias32 = Instance(biasGroup(str))
    bias33 = Instance(biasGroup(str))
    bias34 = Instance(biasGroup(str))
    bias35 = Instance(biasGroup(str))
    bias36 = Instance(biasGroup(str))
    bias37 = Instance(biasGroup(str))
    bias38 = Instance(biasGroup(str))
    bias39 = Instance(biasGroup(str))
    bias40 = Instance(biasGroup(str))
    bias41 = Instance(biasGroup(str))
    bias42 = Instance(biasGroup(str))
    bias43 = Instance(biasGroup(str))
    bias44 = Instance(biasGroup(str))
    bias45 = Instance(biasGroup(str))
    bias46 = Instance(biasGroup(str))
    bias47 = Instance(biasGroup(str))
    bias48 = Instance(biasGroup(str))
    bias49 = Instance(biasGroup(str))
    bias50 = Instance(biasGroup(str))
    bias51 = Instance(biasGroup(str))
    bias52 = Instance(biasGroup(str))
    bias53 = Instance(biasGroup(str))
    bias54 = Instance(biasGroup(str))
    bias55 = Instance(biasGroup(str))
    bias56 = Instance(biasGroup(str))
    bias57 = Instance(biasGroup(str))
    bias58 = Instance(biasGroup(str))
    bias59 = Instance(biasGroup(str))
    bias60 = Instance(biasGroup(str))
    bias61 = Instance(biasGroup(str))
    bias62 = Instance(biasGroup(str))
    bias63 = Instance(biasGroup(str))
    bias64 = Instance(biasBufferGroup(str))
    bias65 = Instance(biasSpecialGroup(str))
    bias66 = Instance(biasSpecialGroup(str))

    #aer configurator
    aerConf = Instance(AerConfiguratorGroup)

    #path configurator
    pathConf = Instance(PathConfiguratorGroup)

    #pci configurator
    FPGA_ID = Range(0,3,1, mode='spinner')
    FPGA_CHNL = Range(1,1,1, mode='spinner')
    pcieconf = Instance(MN256R1Configurator)
    close_pcie = traitsButton()
    status_pcie = Str('undefined')
    pcie_flag = False

    #CMOL
    cmol_enable = traitsButton()
    cmol_view_enabled = False
    
    #tcp socket
    tcp_flag = False
    tcp_flag_not = True
    tcp_ip = Str('127.0.0.1')
    tcp_port = Range(50000,100000,50005)
    open_tcp = traitsButton()
    close_tcp = traitsButton()
    
    #bias tuning
    bias_tune_name = Str('BIAS_NAME_TYPE')
    bias_tune_fine_start = Range(0,255)
    bias_tune_fine_end = Range(0,255)
    tune_repetitions = Range(1,1000)
    tune_pause_ms = Range(0,10000)
    tune_start = traitsButton()

    ##usb flags
    usb_flag = False
    usb_flag_not = True

    #control buttons
    file_to_save = Str
    file_to_load = Str
    save_all_biases = traitsButton()
    load_all_biases = traitsButton()
    load_all_default_biases = traitsButton()
    open_device = traitsButton()
    interface = Enum('PCIe','USB')

    cmol_view = View(Group(Group(Item('bias62', style='custom', show_label=True, label='CMOL_IF_DC_P <->', emphasized=True),
                                   Item('bias12', style='custom', show_label=True, label='CMOL_IF_THR_N <->', emphasized=True),
                                   Item('bias6', style='custom', show_label=True, label='CMOL_IF_AHTAU_N <->', emphasized=True),
                                   Item('bias5', style='custom', show_label=True, label='CMOL_IF_AHW_P <->', emphasized=True),
                                   Item('bias8', style='custom', show_label=True, label='CMOL_IF_TAU_N <->', emphasized=True),
                                   Item('bias2', style='custom', show_label=True, label='CMOL_IF_AHTHR_N <->', emphasized=True),
                                   Item('bias4', style='custom', show_label=True, label='CMOL_IF_RFR_N <->', emphasized=True),
                                   Item('bias11', style='custom', show_label=True, label='CMOL_IF_CASC_N <->', emphasized=True),
                                   label='CMOL_SOMA', show_border=True),
                             Group(Item('bias19', style='custom', show_label=True, label='CMOL_SL_THUP_P <->', emphasized=True),
                                   Item('bias17', style='custom', show_label=True, label='CMOL_SL_WTA_P <->', emphasized=True),
                                   Item('bias15', style='custom', show_label=True, label='CMOL_SL_BUF_N <->', emphasized=True),
                                   Item('bias20', style='custom', show_label=True, label='CMOL_SL_CATAU_P <->', emphasized=True),
                                   Item('bias21', style='custom', show_label=True, label='CMOL_SL_CAW_N <->', emphasized=True),
                                   Item('bias14', style='custom', show_label=True, label='CMOL_SL_MEMTHR_N <->', emphasized=True),
                                   Item('bias18', style='custom', show_label=True, label='CMOL_SL_CATHR_P <->', emphasized=True),
                                   Item('bias13', style='custom', show_label=True, label='CMOL_SL_THDN_P <->', emphasized=True),
                                   Item('bias16', style='custom', show_label=True, label='CMOL_SL_THMIN_N <->', emphasized=True),
                                   label='CMOL_SOMA_LEARN', show_border=True),
                             Group(Item('bias46', style='custom', show_label=True, label='CMOL_WEIGHT_N <->', emphasized=True),
                                   Item('bias24', style='custom', show_label=True, label='CMOL_BUF_BIAS <->', emphasized=True),
                                   Item('bias41', style='custom', show_label=True, label='CMOL_PWLKPOST_P <->', emphasized=True),
                                   Item('bias48', style='custom', show_label=True, label='CMOL_PWLKPRE_P <->', emphasized=True),
                                   Item('bias44', style='custom', show_label=True, label='CMOL_DPI_TAU_P <->', emphasized=True),
                                   Item('bias45', style='custom', show_label=True, label='CMOL_DPI_THR_P <->', emphasized=True),
                                   Item('bias43', style='custom', show_label=True, label='CMOL_PUPREQ_P <->', emphasized=True),
                                   Item('bias63', style='custom', show_label=True, label='CMOL_RSTXARB <->', emphasized=True),
                                   label='CMOL_SPECIAL', show_border=True)))

    traits_view = View(Group(Group(Item('bias0', style='custom', show_label=False),
                                   Item('bias1', style='custom', show_label=False),
                                   Item('bias2', style='custom', show_label=False),
                                   Item('bias3', style='custom', show_label=False),
                                   Item('bias4', style='custom', show_label=False),
                                   Item('bias5', style='custom', show_label=False),
                                   Item('bias6', style='custom', show_label=False),
                                   Item('bias7', style='custom', show_label=False),
                                   Item('bias8', style='custom', show_label=False),
                                   Item('bias9', style='custom', show_label=False),
                                   Item('bias10', style='custom', show_label=False),
                                   Item('bias11', style='custom', show_label=False),
                                   Item('bias12', style='custom', show_label=False),
                                   label='SOMA', show_border=True),
                             Group(Item('bias13', style='custom', show_label=False),
                                   Item('bias14', style='custom', show_label=False),
                                   Item('bias15', style='custom', show_label=False),
                                   Item('bias16', style='custom', show_label=False),
                                   Item('bias17', style='custom', show_label=False),
                                   Item('bias18', style='custom', show_label=False),
                                   Item('bias19', style='custom', show_label=False),
                                   Item('bias20', style='custom', show_label=False),
                                   Item('bias21', style='custom', show_label=False),
                                   label='SOMA_LEARN', show_border=True),
                             Group(Item('bias22', style='custom', show_label=False),
                                   Item('bias23', style='custom', show_label=False),
                                   Item('bias24', style='custom', show_label=False),
                                   Item('bias25', style='custom', show_label=False),
                                   Item('bias26', style='custom', show_label=False),
                                   Item('bias27', style='custom', show_label=False),
                                   label='HOMEOSTATIC', show_border=True),
                             label='Soma'),
                       Group(Group(Item('bias28', style='custom', show_label=False),
                                   Item('bias29', style='custom', show_label=False),
                                   Item('bias30', style='custom', show_label=False),
                                   Item('bias31', style='custom', show_label=False),
                                   Item('bias32', style='custom', show_label=False),
                                   Item('bias33', style='custom', show_label=False),
                                   label='VIRTUAL', show_border=True),
                             Group(Item('bias34', style='custom', show_label=False),
                                   Item('bias35', style='custom', show_label=False),
                                   Item('bias36', style='custom', show_label=False),
                                   Item('bias37', style='custom', show_label=False),
                                   Item('bias38', style='custom', show_label=False),
                                   Item('bias39', style='custom', show_label=False),
                                   Item('bias40', style='custom', show_label=False),
                                   Item('bias41', style='custom', show_label=False),
                                   Item('bias42', style='custom', show_label=False),
                                   Item('bias43', style='custom', show_label=False),
                                   Item('bias44', style='custom', show_label=False),
                                   Item('bias45', style='custom', show_label=False),
                                   label='PLASTIC', show_border=True),
                             Group(Item('bias46', style='custom', show_label=False),
                                   Item('bias47', style='custom', show_label=False),
                                   Item('bias48', style='custom', show_label=False),
                                   Item('bias49', style='custom', show_label=False),
                                   Item('bias50', style='custom', show_label=False),
                                   Item('bias51', style='custom', show_label=False),
                                   Item('bias52', style='custom', show_label=False),
                                   Item('bias53', style='custom', show_label=False),
                                   Item('bias54', style='custom', show_label=False),
                                   Item('bias55', style='custom', show_label=False),
                                   Item('bias56', style='custom', show_label=False),
                                   Item('bias57', style='custom', show_label=False),
                                   label='NON_PLASTIC', show_border=True),
                             label='Synapses'),
                       Group(Group(Item('bias58', style='custom', show_label=False),
                                   Item('bias59', style='custom', show_label=False),
                                   Item('bias60', style='custom', show_label=False),
                                   Item('bias61', style='custom', show_label=False),
                                   Item('bias62', style='custom', show_label=False),
                                   Item('bias63', style='custom', show_label=False),
                                   label='ADDITIONAL', show_border=True),
                             Group(Item('bias64', style='custom', show_label=False),
                                   Item('bias65', style='custom', show_label=False),
                                   Item('bias66', style='custom', show_label=False),
                                   label='SPECIAL', show_border=True),
                             label='Special'),
                       Group(Item(name='file_to_load'),
                             Item('load_all_biases', label='Load File', show_label=False),
                             Item('load_all_default_biases', label='Load Default', show_label=False),
                             label='Load'),
                       Group(Item(name='file_to_save'),
                             Item('save_all_biases', label='Save File', show_label=False),
                             label='Save'),
                       Group(Item('aerConf', style='custom', show_label=False),
                             label='Chip Configurator'),
                       Group(Item('pathConf', style='custom', show_label=False),
                             label='Path Configurator'),
                       Group(Group(Spring(springy=True),
                                   Item(name='interface', label='Please choose an interface'),
                                   Item('open_device', label='Open Device', show_label=False),
                                   label='Interface', show_border=True, orientation='horizontal'),
                             Group(Item(name='FPGA_ID', label='Device ID', enabled_when='pcie_flag_not'),
                                   Item(name='FPGA_CHNL', label='Channel', enabled_when='pcie_flag_not'),
                                   Spring(springy=True),
                                   Item(name='status_pcie', label='Status', style='readonly', emphasized=True),
                                   Item(name='close_pcie', label='Close', show_label=False, enabled_when='pcie_flag'),
                                   label='PCIe', show_border=True, orientation='horizontal'),
                             Group(Item(name='tcp_ip', label='IP Address', enabled_when='tcp_flag_not'),
                                   Item(name='tcp_port', label='Port', enabled_when='tcp_flag_not'),
                                   Spring(springy=True),
                                   Item('open_tcp', label='Connect', show_label=False, enabled_when='tcp_flag_not'),
                                   Item('close_tcp', label='Close', show_label=False, enabled_when='tcp_flag'),
                                   label='TCP/IP', show_border=True, orientation='horizontal'),
                             Group(Item(name='bias_tune_name', style='readonly', show_label=False, emphasized=True),
                                   Spring(springy=True),
                                   Item(name='bias_tune_fine_start',label='Fine current from'),
                                   Item(name='bias_tune_fine_end',label='to'),
                                   Spring(springy=True),
                                   Item(name='tune_repetitions', label='Repetitions'),
                                   Item(name='tune_pause_ms', label='Pause (ms)'),
                                   Item(name='tune_start',label='Start',show_label=False),
                                   label='Bias Tuning', show_border=True, orientation='horizontal'),
                             Group(Spring(springy=True),
                                   Item(name='cmol_enable',label='Open CMOL biases control dialog',show_label=False),
                                   orientation='horizontal',show_border=True,label='CMOL'),
                             label='Setup'))

    def _cmol_enable_fired(self):
        if (not(self.cmol_view_enabled)):
            self.configure_traits(view='cmol_view')
            self.cmol_view_enabled = True

    def _program_monitor_fired(self):
        return
        
    def _open_tcp_fired(self):
        if self.tcp_flag_not:
            self.tcp_flag = True
            self.tcp_flag_not = False
            configurator.openTCP(self.tcp_ip,self.tcp_port)
            
    def _close_tcp_fired(self):
        if self.tcp_flag:
            self.tcp_flag = False
            self.tcp_flag_not = True
            configurator.closeTCP()
            
    def _tune_start_fired(self):
        #change bias here
        self.bias_tune_name = self.bias12.name
        for r in range(0,self.tune_repetitions):
            for i in range(self.bias_tune_fine_start,self.bias_tune_fine_end+1):
                self.bias12.fine = i
                time.sleep(self.tune_pause_ms/1000.0)
                print 'Repetition: ' + str(r)

    def openDefaultDevice(self):
        #configurator.openPCIe(self.FPGA_ID,self.FPGA_CHNL)
        self.status_pcie = 'closed'
        self.pcie_flag = False
        self.pcie_flag_not = True

    def _open_device_fired(self):
        if self.interface == 'PCIe':
            if self.pcie_flag == False:
                configurator.openPCIe(self.FPGA_ID,self.FPGA_CHNL)
                self.status_pcie = 'opened'
                self.pcie_flag = True
                self.pcie_flag_not = False
        if self.interface == 'USB':
            print "usb interface selected"
            import biasusb_wrap
            self.usb_flag = True
            self.usb_flag_not = False

    def _close_pcie_fired(self):
        if self.pcie_flag == True:
            configurator.closePCIe()
            self.status_pcie = 'closed'
            self.pcie_flag = False
            self.pcie_flag_not = True

    def _save_all_biases_fired(self):
        output = ''
        output += self.bias0.exportToString()+'\n'
        output += self.bias1.exportToString()+'\n'
        output += self.bias2.exportToString()+'\n'
        output += self.bias3.exportToString()+'\n'
        output += self.bias4.exportToString()+'\n'
        output += self.bias5.exportToString()+'\n'
        output += self.bias6.exportToString()+'\n'
        output += self.bias7.exportToString()+'\n'
        output += self.bias8.exportToString()+'\n'
        output += self.bias9.exportToString()+'\n'
        output += self.bias10.exportToString()+'\n'
        output += self.bias11.exportToString()+'\n'
        output += self.bias12.exportToString()+'\n'
        output += self.bias13.exportToString()+'\n'
        output += self.bias14.exportToString()+'\n'
        output += self.bias15.exportToString()+'\n'
        output += self.bias16.exportToString()+'\n'
        output += self.bias17.exportToString()+'\n'
        output += self.bias18.exportToString()+'\n'
        output += self.bias19.exportToString()+'\n'
        output += self.bias20.exportToString()+'\n'
        output += self.bias21.exportToString()+'\n'
        output += self.bias22.exportToString()+'\n'
        output += self.bias23.exportToString()+'\n'
        output += self.bias24.exportToString()+'\n'
        output += self.bias25.exportToString()+'\n'
        output += self.bias26.exportToString()+'\n'
        output += self.bias27.exportToString()+'\n'
        output += self.bias28.exportToString()+'\n'
        output += self.bias29.exportToString()+'\n'
        output += self.bias30.exportToString()+'\n'
        output += self.bias31.exportToString()+'\n'
        output += self.bias32.exportToString()+'\n'
        output += self.bias33.exportToString()+'\n'
        output += self.bias34.exportToString()+'\n'
        output += self.bias35.exportToString()+'\n'
        output += self.bias36.exportToString()+'\n'
        output += self.bias37.exportToString()+'\n'
        output += self.bias38.exportToString()+'\n'
        output += self.bias39.exportToString()+'\n'
        output += self.bias40.exportToString()+'\n'
        output += self.bias41.exportToString()+'\n'
        output += self.bias42.exportToString()+'\n'
        output += self.bias43.exportToString()+'\n'
        output += self.bias44.exportToString()+'\n'
        output += self.bias45.exportToString()+'\n'
        output += self.bias46.exportToString()+'\n'
        output += self.bias47.exportToString()+'\n'
        output += self.bias48.exportToString()+'\n'
        output += self.bias49.exportToString()+'\n'
        output += self.bias50.exportToString()+'\n'
        output += self.bias51.exportToString()+'\n'
        output += self.bias52.exportToString()+'\n'
        output += self.bias53.exportToString()+'\n'
        output += self.bias54.exportToString()+'\n'
        output += self.bias55.exportToString()+'\n'
        output += self.bias56.exportToString()+'\n'
        output += self.bias57.exportToString()+'\n'
        output += self.bias58.exportToString()+'\n'
        output += self.bias59.exportToString()+'\n'
        output += self.bias60.exportToString()+'\n'
        output += self.bias61.exportToString()+'\n'
        output += self.bias62.exportToString()+'\n'
        output += self.bias63.exportToString()+'\n'
        output += self.bias64.exportToString()+'\n'
        output += self.bias65.exportToString()+'\n'
        output += self.bias66.exportToString()+'\n'
        configurator.save_all_biases(self.file_to_save,output)
        return

    def _load_all_default_biases_fired(self):
        self.load_biases('')

    def _load_all_biases_fired(self):
        self.load_biases(self.file_to_load)

    def load_biases(self,file):
        self.bias0.setDefaults(file)
        self.bias1.setDefaults(file)
        self.bias2.setDefaults(file)
        self.bias3.setDefaults(file)
        self.bias4.setDefaults(file)
        self.bias5.setDefaults(file)
        self.bias6.setDefaults(file)
        self.bias7.setDefaults(file)
        self.bias8.setDefaults(file)
        self.bias9.setDefaults(file)
        self.bias10.setDefaults(file)
        self.bias11.setDefaults(file)
        self.bias12.setDefaults(file)
        self.bias13.setDefaults(file)
        self.bias14.setDefaults(file)
        self.bias15.setDefaults(file)
        self.bias16.setDefaults(file)
        self.bias17.setDefaults(file)
        self.bias18.setDefaults(file)
        self.bias19.setDefaults(file)
        self.bias20.setDefaults(file)
        self.bias21.setDefaults(file)
        self.bias22.setDefaults(file)
        self.bias23.setDefaults(file)
        self.bias24.setDefaults(file)
        self.bias25.setDefaults(file)
        self.bias26.setDefaults(file)
        self.bias27.setDefaults(file)
        self.bias28.setDefaults(file)
        self.bias29.setDefaults(file)
        self.bias30.setDefaults(file)
        self.bias31.setDefaults(file)
        self.bias32.setDefaults(file)
        self.bias33.setDefaults(file)
        self.bias34.setDefaults(file)
        self.bias35.setDefaults(file)
        self.bias36.setDefaults(file)
        self.bias37.setDefaults(file)
        self.bias38.setDefaults(file)
        self.bias39.setDefaults(file)
        self.bias40.setDefaults(file)
        self.bias41.setDefaults(file)
        self.bias42.setDefaults(file)
        self.bias43.setDefaults(file)
        self.bias44.setDefaults(file)
        self.bias45.setDefaults(file)
        self.bias46.setDefaults(file)
        self.bias47.setDefaults(file)
        self.bias48.setDefaults(file)
        self.bias49.setDefaults(file)
        self.bias50.setDefaults(file)
        self.bias51.setDefaults(file)
        self.bias52.setDefaults(file)
        self.bias53.setDefaults(file)
        self.bias54.setDefaults(file)
        self.bias55.setDefaults(file)
        self.bias56.setDefaults(file)
        self.bias57.setDefaults(file)
        self.bias58.setDefaults(file)
        self.bias59.setDefaults(file)
        self.bias60.setDefaults(file)
        self.bias61.setDefaults(file)
        self.bias62.setDefaults(file)
        self.bias63.setDefaults(file)
        self.bias64.setDefaults(file)
        self.bias65.setDefaults(file)
        self.bias66.setDefaults(file)


#biasProgrammer object
configurator = MN256R1Configurator()

#init values
def_biases = biasTraits(aerConf=AerConfiguratorGroup(configurator),
                        pathConf=PathConfiguratorGroup(configurator),
                        pcieconf=configurator,
                        bias0=biasGroup('IF_RST_N'),
                        bias1=biasGroup('IF_BUF_P'),
                        bias2=biasGroup('IF_ATHR_N'),
                        bias3=biasGroup('IF_RFR1_N'),
                        bias4=biasGroup('IF_RFR2_N'),
                        bias5=biasGroup('IF_AHW_P'),
                        bias6=biasGroup('IF_AHTAU_N'),
                        bias7=biasGroup('IF_DC_P'),
                        bias8=biasGroup('IF_TAU2_N'),
                        bias9=biasGroup('IF_TAU1_N'),
                        bias10=biasGroup('IF_NMDA_N'),
                        bias11=biasGroup('IF_CASC_N'),
                        bias12=biasGroup('IF_THR_N'),
                        bias13=biasGroup('SL_THDN_P'),
                        bias14=biasGroup('SL_MEMTHR_N'),
                        bias15=biasGroup('SL_BUF_N'),
                        bias16=biasGroup('SL_THMIN_N'),
                        bias17=biasGroup('SL_WTA_P'),
                        bias18=biasGroup('SL_CATHR_P'),
                        bias19=biasGroup('SL_THUP_P'),
                        bias20=biasGroup('SL_CATAU_P'),
                        bias21=biasGroup('SL_CAW_N'),
                        bias22=biasGroup('FB_REF_P'),
                        bias23=biasGroup('FB_WTA_N'),
                        bias24=biasGroup('FB_BUF_P'),
                        bias25=biasGroup('FB_CASC_N'),
                        bias26=biasGroup('FB_INVERSE_TAIL_N'),
                        bias27=biasGroup('FB_INVERSE_REF_N'),
                        bias28=biasGroup('VA_INH_P'),
                        bias29=biasGroup('VDPII_TAU_N'),
                        bias30=biasGroup('VDPII_THR_N'),
                        bias31=biasGroup('VA_EXC_N'),
                        bias32=biasGroup('VDPIE_TAU_P'),
                        bias33=biasGroup('VDPIE_THR_P'),
                        bias34=biasGroup('PA_WDRIFTDN_N'),
                        bias35=biasGroup('PA_WDRIFTUP_P'),
                        bias36=biasGroup('PA_DELTAUP_P'),
                        bias37=biasGroup('PA_DELTADN_N'),
                        bias38=biasGroup('PA_WHIDN_N'),
                        bias39=biasGroup('PA_WTHR_P'),
                        bias40=biasGroup('PA_WDRIFT_P'),
                        bias41=biasGroup('PA_PWLK_P'),
                        bias42=biasGroup('PDPI_BUF_N'),
                        bias43=biasGroup('PDPI_VMONPU_P'),
                        bias44=biasGroup('PDPI_TAU_P'),
                        bias45=biasGroup('PDPI_THR_P'),
                        bias46=biasGroup('NPA_WEIGHT_STD_N'),
                        bias47=biasGroup('NPA_WEIGHT_INH0_N'),
                        bias48=biasGroup('NPA_PWLK_P'),
                        bias49=biasGroup('NPA_WEIGHT_INH1_N'),
                        bias50=biasGroup('NPA_WEIGHT_EXC_P'),
                        bias51=biasGroup('NPA_WEIGHT_EXC1_P'),
                        bias52=biasGroup('NPA_WEIGHT_EXC0_P'),
                        bias53=biasGroup('NPA_WEIGHT_INH_N'),
                        bias54=biasGroup('NPDPIE_THR_P'),
                        bias55=biasGroup('NPDPIE_TAU_P'),
                        bias56=biasGroup('NPDPII_TAU_P'),
                        bias57=biasGroup('NPDPII_THR_P'),
                        bias58=biasGroup('BIAS_58'),
                        bias59=biasGroup('BIAS_59'),
                        bias60=biasGroup('BIAS_60'),
                        bias61=biasGroup('BIAS_61'),
                        bias62=biasGroup('BIAS_62'),
                        bias63=biasGroup('BIAS_63'),
                        bias64=biasBufferGroup('BUFFER_BIASES'),
                        bias65=biasSpecialGroup('SSP'),
                        bias66=biasSpecialGroup('SSN'),
                        file_to_save='biases/',
                        file_to_load='biases/',
                        name='mn256r1')

def_biases.openDefaultDevice()
def_biases.load_biases('')

def_biases.configure_traits()
