# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#Enthougt and Chaco imports
from traits.api import HasTraits, Instance, Int, CFloat, Float, List, Enum,\
    Trait, Callable, Range, Bool
from traitsui.api import View, Item, Group, Handler, RangeEditor
from traitsui.api import Image as traitsImage
from traitsui.api import ImageEditor as traitsImageEditor
from traits.api import Button as traitsButton
from chaco.api import Plot, ArrayPlotData, jet, Greys
from chaco.default_colormaps import *
from enable.component_editor import ComponentEditor
from traitsui.menu import Action, CloseAction, MenuBar, Menu, LiveButtons

#Other imports
from numpy import exp, linspace, meshgrid, append
import numpy
import Queue

import wx

#Imports for AER monitoring
import pyAex
from pyNCS.pyST import *
import pyNCS

from expSetup_monster_learning import *

from poissoninp import PoissonStimulator

# teacher signal on 1st excitatory synapse of neuron 0
SYN_PADDR = 2228224

class EventsUpdater():

        def __init__(self, gui, host='localhost', port=50001, channel=0,
                     fps=25):
                self.gui = gui
                self.port = port
                self.host = host
                self.fps = fps
                self.channel = channel
                self.tDuration = 5.
                self.gui.updater = self
                self.stcs=getDefaultMonChannelAddress()
                #addrBuildHashTable(self.stcs[channel])
                self.eventsQueue = nsetup.com_api.aexclient.AEXMonClient(MonChannelAddress=self.stcs,
                                                           channels = [self.channel],
                                                           host = self.host,
                                                           port = self.port,
                                                           autostart = True,
                                                           fps = self.fps)
                self.neurons = numpy.array([])
                self.times = numpy.array([])
                
                self.tot_neurons = numpy.array([])
                self.tot_times = numpy.array([])
                                
                self.gui.channel = channel
                
                #count for updating the rate
                self.clockcount = 0
                self.ratecount = 0
                self.cv = 0
                
        def fetch(self,*args):
                '''
                The function fetches the data from the server and updates the plot (calles the 'update' function)
                '''
                tDuration = self.gui.tDuration
                
                cur_neurons = numpy.array([])
                cur_times = numpy.array([])
                #for now i assume that the reading of que is faster than writing.. something more intelligent can go here.
                while True:
                        try:
                                eventsPacket = self.eventsQueue.buffer.get(block=False)
                                cur_neurons = append(cur_neurons, eventsPacket.get_ad(), axis = 0,)
                                cur_times = append(cur_times, eventsPacket.get_tm())                
                                                        
                        except Queue.Empty, e:
                                break
                
                self.neurons = cur_neurons
                self.times = cur_times
                                
                old_neurons = self.tot_neurons
                old_times = self.tot_times
                
                new_times = append(old_times,cur_times*1e-6)
                new_neurons = append(old_neurons,cur_neurons)
                
                try:
                        ind = new_times.searchsorted(new_times[-1]-tDuration,side='right')
                        self.tot_neurons = new_neurons[ind:]
                        self.tot_times = new_times[ind:]
                except IndexError, e:
                        pass
                        #print('Warning:Index Error .. (No data)')
                
                
                
                #update clock count
                self.clockcount += 1
                
                self.update()
                
                
        def update(self):
                '''
                update() function updates all the components.
                The class can be inherited and then the update function can be overwritten to make custom filters.
                '''
                self.updatePlot()
                self.updateMeanrate()
                self.updateCV()
        
        
        def updateMeanrate(self):
                self.ratecount += len(self.times)
                if self.clockcount % int(self.fps/2) == 0:
                        self.gui.meanrate = self.ratecount*2 #update the display
                        self.ratecount = 0 #reset the counter
        
        def updateCV(self):
                if len(self.times) < 1:
                    self.cv = -1
                    return
                d = diff(self.times)
                self.cv = std(d)/average(d)
                if self.clockcount % int(self.fps/2) == 0:
                        self.gui.cv = self.cv #update the display
        
        def updatePlot(self):
                '''
                updatePlot() function updates the plot
                '''
                try:
                        self.gui.plotdata.set_data('neurons',self.tot_neurons)
                        self.gui.plotdata.set_data('times', self.tot_times)
                        self.gui.plot.request_redraw()
                except IndexError, e:
                        print('Warning:Index Error .. (No data)')
                        
                return True
        
                
        def stop(self):
                self.eventsQueue.stop()
                
        def __del__(self):
                self.eventsQueue.stop()

class Controller(Handler):
        
        view = Instance(HasTraits)
        
        def init(self, info):
                self.view = info.object        
        
        def edit_plot(self, ui_info):
                self.view.configure_traits(view="plot_edit_view")

        def load_default_biases(self, ui_info):
            nsetup.chips['mn256r1'].load_parameters('biases/biases_learning_interface.biases')

        #def setscanx_0(self, ui_info):
            #chip.setscanx(0)

class LearningGUI(HasTraits):
        plot = Instance(Plot)
        meanrate = CFloat(0.0)
        cv = CFloat(0.0)
        teach = Bool(0)
        teach_active = Bool(0)
        teach_high = Range(100., 1000.)
        teach_low = Range(100., 1000.)
        f_max = Range(0., 1000.)
        f_min = Range(0., 1000.)
        updater = Instance(EventsUpdater)
        npre = Int(1)
        f_max_demo = Range(0., 1000.)
        durationl = Range(0., 10000.)
        
        stimlearn = traitsButton()
        inilearn = traitsButton()
        set_wij = traitsButton()
        get_wij = traitsButton()
        reset0_wij = traitsButton()
        reset1_wij = traitsButton()
        learning = Bool

        #Plot properties
        tDuration = Range(0.,20) 
        channel = Enum(range(getDefaultMonChannelAddress().nChannels))
        
        period = 1
        port = 50002
        host = nsetup.communicator.kwargs['host']
        stimulator = PoissonStimulator(SeqChannelAddress=None,
                                            channel=1,
                                            host=host,
                                            port_stim=port,
                                            rates=[],
                                            period=period)

        def _tDuration_default(self): return 5.
        
        def _meanrate_default(self): return 0.
        
        def _cv_default(self): return 0.
        
        def _learning_default(self): return True
        
        def _teach_high_default(self): return 200.
        def _teach_low_default(self): return 500.
        def _f_max_default(self): return 55.
        def _f_min_default(self): return 10.
        def _f_max_demo_default(self): return 55.
        def _durationl_default(self): return 200.
        
        def __init__(self, markersize=4, marker='circle', color='black'):
                super(LearningGUI, self).__init__()
                self.plotdata = ArrayPlotData(neurons = [0], times = [0])
                plot = Plot(self.plotdata)
                plot.plot(( "times", "neurons" ), 
                          type = "scatter",
                          marker = marker,
                          marker_size = markersize,
                          color=color)
                self.plot = plot
                
                self.wijstate = ArrayPlotData(imagedata=zeros((124, 28)))
                wijstate_plot = Plot(self.wijstate)
                wijstate_plot.img_plot("imagedata")
                self.wijstate_plot = wijstate_plot
        
                self.char = ArrayPlotData(imagedata=zeros((28, 28)))
                char_plot = Plot(self.char)
                char_plot.img_plot("imagedata")
                self.char_plot = char_plot
        
        def _channel_changed(self):
                print('Switching to channel: %d' % self.channel)
                self.updater.channel = self.channel
                self.updater.tot_neurons = []
                self.updater.tot_times = []
                try:
                        self.updater.eventsQueue.stop()
                except:
                        pass
                #addrBuildHashTable(self.updater.stcs[self.channel])
                self.updater.eventsQueue = pyAex.netMonClient(MonChannelAddress=self.updater.stcs,
                                                           channels = [self.channel],
                                                           host = self.updater.host,
                                                           port = self.updater.port,
                                                           autostart = True,
                                                           fps = self.updater.fps)


        def _teach_active_changed(self):
            if self.teach_active:
                host = nsetup.communicator.kwargs['host']
                self.stimulator = PoissonStimulator(SeqChannelAddress=None,
                                                    channel=1,
                                                    seq_export=False,
                                                    host=host,
                                                    port_stim=50002,
                                                    rates=[[SYN_PADDR,
                                                            self.teach_low]],
                                                    period=self.period)
                self.set_teacher()
                self.stimulator.start()
            else:
                self.stimulator.stop()

        def _teach_changed(self):
            self.set_teacher()

        def set_teacher(self):
            if self.teach:
                self.stimulator.rates = [[SYN_PADDR,
                                          self.teach_high]]
            else:
                self.stimulator.rates = [[SYN_PADDR,
                                          self.teach_low]]
        
        def _teach_high_changed(self):
            if self.teach:
                self.stimulator.rates = [[SYN_PADDR,
                                          self.teach_high]]
        
        def _teach_low_changed(self):
            if not self.teach:
                self.stimulator.rates = [[SYN_PADDR,
                                          self.teach_low]]

        def _learning_changed(self):
            if self.learning:
                chip.setBias(learning_on_biases)
            else: 
                chip.setBias(learning_off_biases)

        traits_view = View(
            Group(
                Group(
                    Group(
                        Item('teach_active', label='Teacher active   '),
                        Group(
                            Item('teach_low', label='NO freq   ',
                                 editor=RangeEditor(low=100, high=1000, mode='xslider')),
                            Item('teach_high', label='YES freq   ',
                                 editor=RangeEditor(low=100, high=1000, mode='xslider')),
                            orientation='vertical'
                        ),
                        Item('teach', label='Teach!   '),
                        label='Teacher',
                        orientation='vertical',
                    ),
                    #Group(
                        #Item('f_max_demo', label="f_max   "),
                        #Item('npre', label="Rounds   "),
                        #Item('durationl', label="Stim duration   "),
                        #Item('inilearn', show_label=False),
                        #label='Demo stimulator',
                        #orientation='vertical',
                    #),
                    Group(
                        # can't resize to fit in the window
                        #Item('char_plot', editor=ComponentEditor(), show_label=False),
                        Item('f_max', label="f_max   "),
                        Item('f_min', label="f_min   "),
                        Item('durationl', label="Stim duration   "),
                        Item('learning', label="Plasticity   "),
                        Item('stimlearn', show_label=False),
                        label='Stimulator',
                        orientation='vertical',
                    ),
                    Group(
                        Item('meanrate', label='MeanRate(Hz)    ', style='readonly'),
                        Item('cv', label='ISI CV    ', style='readonly'),
                        label='Quick measures',
                        orientation='vertical'
                    ),
                ),
                Group(
                    Item('plot', editor=ComponentEditor(), show_label=False),
                    label='Viewer',
                    orientation='vertical',
                ),
                orientation='horizontal'
            ),
            dock='tab',
            menubar=MenuBar(Menu(Action(name="Edit Plot",
                                        action="edit_plot"),
                                 CloseAction,
                                 name="File"),
                            Menu(Action(name="Default",
                                        action="load_default_biases"),
                                 Action(name="Set scan 0",
                                        action="setscanx_0"),
                                 name="Biases")),
            buttons=LiveButtons,
            handler = Controller,
            width=1600, 
            height=600, 
            resizable=True, 
            title="Leonardo",
        )
        
        plot_edit_view = View(Group(Item('tDuration'),
                                    #Item('colormap'),
                                    Item('channel')),
                              buttons=['OK','Cancel'])
