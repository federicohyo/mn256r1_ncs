# dvspair gui
# author: Marc Osswald - marc@ini.phys.ethz.ch
# Mar 2014

from traits.api import HasTraits, Str, Int, Range, Array, List, File, Float, Enum, Instance, Bool
from traitsui.api import View, Item, Group, RangeEditor, Spring
import traitsui
from traits.api import Button as traitsButton

# Import libraries
from pylab import *
import numpy as np
from DVSPAIRConfigurator import DVSPAIRConfigurator


class dvspairTraits(HasTraits):

    #pci configurator
    FPGA_ID = Range(0,3,0, mode='spinner')
    FPGA_CHNL = Range(0,0,0, mode='spinner')
    pcieconf = Instance(DVSPAIRConfigurator)
    close_pcie = traitsButton()
    status_pcie = Str('undefined')
    pcie_flag = False

    #control buttons
    open_device = traitsButton()
    interface = Enum('PCIe','Serial')

    #filter settings
    leftYhigh = Range(0,127,127)
    leftYlow = Range(0,127,0)
    leftXhigh = Range(0,127,127)
    leftXlow = Range(0,127,0)
    rightYhigh = Range(0,127,127)
    rightYlow = Range(0,127,0)
    rightXhigh = Range(0,127,127)
    rightXlow = Range(0,127,0)

    #filter configuration
    ponEnable = Bool(True)
    poffEnable = Bool(True)
    fx2Enable = Bool(True)
    pcieEnable = Bool(True)
    leftEnable = Bool(True)
    rightEnable = Bool(True)


    traits_view = View(Group(Group(Item(name='leftXlow', label='Xmin'),
                                   Item(name='leftXhigh', label='Xmax'),
                                   Item(name='leftYlow', label='Ymin'),
                                   Item(name='leftYhigh', label='Ymax'),
                                   label='Left Boundaries', orientation='horizontal', show_border=True),
                             Group(Item(name='rightXlow', label='Xmin'),
                                   Item(name='rightXhigh', label='Xmax'),
                                   Item(name='rightYlow', label='Ymin'),
                                   Item(name='rightYhigh', label='Ymax'),
                                   label='Right Boundaries', orientation='horizontal', show_border=True),
                             Group(Item(name='ponEnable', label='Polarity ON'),
                                   Item(name='poffEnable', label='Polarity OFF'),
                                   Item(name='fx2Enable', label='To FX2'),
                                   Item(name='pcieEnable', label='To PCIe'),
                                   Item(name='leftEnable', label='Left DVS'),
                                   Item(name='rightEnable', label='Right DVS'),
                                   label='Configuration', orientation='horizontal', show_border=True),
                             label='Filter'),
                       Group(Group(Spring(springy=True),
                                   Item(name='interface', label='Please choose an interface'),
                                   Item('open_device', label='Open Device', show_label=False),
                                   label='Interface', show_border=True, orientation='horizontal'),
                             Group(Item(name='FPGA_ID', label='Device ID', enabled_when='pcie_flag_not'),
                                   Item(name='FPGA_CHNL', label='Channel', enabled_when='pcie_flag_not'),
                                   Spring(springy=True),
                                   Item(name='status_pcie', label='Status', style='readonly', emphasized=True),
                                   Item(name='close_pcie', label='Close', show_label=False, enabled_when='pcie_flag'),
                                   label='PCIe Sequencer', show_border=True, orientation='horizontal'),
                             label='Setup'))

    def _leftXlow_changed(self):
        self.setLeftFilterParams()

    def _leftXhigh_changed(self):
        self.setLeftFilterParams()

    def _leftYlow_changed(self):
        self.setLeftFilterParams()

    def _leftYhigh_changed(self):
        self.setLeftFilterParams()

    def setLeftFilterParams(self):
        bits = (int(self.leftXhigh)<<21) + (int(self.leftXlow)<<14) + (int(self.leftYhigh)<<7) + (int(self.leftYlow)<<0)
        self.pcieconf.send_left_params(bits)

    def _rightXlow_changed(self):
        self.setRightFilterParams()

    def _rightXhigh_changed(self):
        self.setRightFilterParams()

    def _rightYlow_changed(self):
        self.setRightFilterParams()

    def _rightYhigh_changed(self):
        self.setRightFilterParams()

    def setRightFilterParams(self):
        bits = (int(self.rightXhigh)<<21) + (int(self.rightXlow)<<14) + (int(self.rightYhigh)<<7) + (int(self.rightYlow)<<0)
        self.pcieconf.send_right_params(bits)

    def _ponEnable_changed(self):
        self.setConfParams()

    def _poffEnable_changed(self):
        self.setConfParams()

    def _fx2Enable_changed(self):
        self.setConfParams()

    def _pcieEnable_changed(self):
        self.setConfParams()

    def _leftEnable_changed(self):
        self.setConfParams()

    def _rightEnable_changed(self):
        self.setConfParams()

    def setConfParams(self):
        bits = (int(self.poffEnable)<<5) + (int(self.ponEnable)<<4) + (int(self.fx2Enable)<<3) + (int(self.pcieEnable)<<2) + (int(self.leftEnable)<<1) + (int(self.rightEnable))
        self.pcieconf.send_conf_params(bits)

    def openDefaultDevice(self):
        self.pcieconf.openPCIe(self.FPGA_ID,self.FPGA_CHNL)
        self.status_pcie = 'opened'
        self.pcie_flag = True
        self.pcie_flag_not = False

    def _open_device_fired(self):
        if self.interface == 'PCIe':
            if self.pcie_flag == False:
                self.pcieconf.openPCIe(self.FPGA_ID,self.FPGA_CHNL)
                self.status_pcie = 'opened'
                self.pcie_flag = True
                self.pcie_flag_not = False

    def _close_pcie_fired(self):
        if self.pcie_flag == True:
            self.pcieconf.closePCIe()
            self.status_pcie = 'closed'
            self.pcie_flag = False
            self.pcie_flag_not = True


#dvs configurator object
configurator = DVSPAIRConfigurator()

#init values
def_dvspair = dvspairTraits(pcieconf=configurator, name='dvspair')

def_dvspair.openDefaultDevice()
def_dvspair.setConfParams()
def_dvspair.setLeftFilterParams()
def_dvspair.setRightFilterParams()

def_dvspair.configure_traits()
