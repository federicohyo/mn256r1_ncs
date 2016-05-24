__author__ = 'marc'

from traits.api import HasTraits, Str, Int, Range, Array, List, File, Float, Enum, Instance, Bool
from traitsui.api import View, Item, Group, RangeEditor, Spring
import traitsui
from traits.api import Button as traitsButton
from MN256R1Configurator import MN256R1Configurator
import time

# Import libraries
from pylab import *
import numpy as np

class PathConfiguratorGroup(HasTraits):


    configurator = MN256R1Configurator

    # Path
    enableAerPCIEToMonitor = Bool
    inputSelect = Bool
    outputSelect = Bool
    enablePipeline = Bool
    defaultConf = traitsButton()

    view = View(Group(Group(Group(Item(name='enablePipeline',label='Enable pipeline'),
                                  Item(name='outputSelect',label='Enable auto output'),
                                  Item(name='inputSelect',label='Enable auto input'),
                                  Item(name='enableAerPCIEToMonitor',label='Enable PCIe to FX2'),
                                  Spring(springy=True),
                                  Item(name='defaultConf', label='Default', show_label=False),
                                  orientation='horizontal'),
                            show_border=True,
                            label='Path'),
                      orientation='vertical'))

    def __init__(self, conf):
        self.configurator = conf

    def _defaultConf_fired(self):
        self.enableAerPCIEToMonitor = False
        self.inputSelect = False
        self.outputSelect = False
        self.enablePipeline = False

    def _enableAerPCIEToMonitor_changed(self):
        if self.enableAerPCIEToMonitor==True:
            self.configurator.set_conf(1)
        else:
            self.configurator.set_conf(0)

    def _inputSelect_changed(self):
        if self.inputSelect==True:
            self.configurator.set_conf(3)
        else:
            self.configurator.set_conf(2)

    def _outputSelect_changed(self):
        if self.outputSelect==True:
            self.configurator.set_conf(5)
        else:
            self.configurator.set_conf(4)

    def _enablePipeline_changed(self):
        if self.enablePipeline==True:
            self.configurator.set_conf(7)
        else:
            self.configurator.set_conf(6)