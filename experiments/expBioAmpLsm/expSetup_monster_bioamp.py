#!/usr/env/python
import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('../api/wij/')
sys.path.append('../api/bioamp/')
sys.path.append('../api/perceptrons/')
import sys
sys.path.append('/home/federico/projects/work/trunk/code/python/spkInt/scripts/')
import functions
sys.path.append('../api/reservoir/')
sys.path.append('../api/retina/')
sys.path.append('../gui/reservoir_display/')
from scipy import interpolate
import reservoir as L

from perceptrons import Perceptrons
from wij import SynapsesLearning
from bioamp import Bioamp
import time
import scipy.signal
import subprocess


res = L.Reservoir() #reservoir without configuring the chip 
    
    
######################################
# Configure chip
try:
  is_configured
except NameError:
  print "Configuring chip"
  is_configured = False
else:
  print "Chip is configured: ", is_configured

scope_capture = False
ckeck_variability_input = False
one_neuron_simple_exp = False
perceptrons_experiment = False
population_coding = True

if (is_configured == False):  
    #prefix='../'
    #setuptype = '../setupfiles/mc_final_mn256r1.xml'
    #setupfile = '../setupfiles/final_mn256r1_retina_monster.xml'
    prefix='../'  
    setuptype = '../setupfiles/mc_final_mn256r1_adcs.xml'
    setupfile = '../setupfiles/final_mn256r1_adcs.xml'
    nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
    chip = nsetup.chips['mn256r1']

    p = pyNCS.Population('', '')
    p.populate_all(nsetup, 'mn256r1', 'excitatory')
    
    inputpop = pyNCS.Population('','')
    inputpop.populate_by_id(nsetup,'mn256r1', 'excitatory', np.linspace(0,255,256))  
    
    #reset multiplexer
    bioamp = Bioamp(inputpop)
    bioamp._init_fpga_mapper()
    
    
    matrix_b = np.random.choice([0,0,1],[256,256])
    matrix_e_i = np.random.choice([0,0,1,1,1],[256,256])
    index_w_1 = np.where(matrix_b == 1)
    matrix_weight = np.zeros([256,256])
    matrix_weight[index_w_1] = 1
    index_w_2 = np.where(matrix_weight != 1)
    matrix_weight[index_w_2] = 2
    
    matrix_recurrent = np.random.choice([0,0,1,1,1],[256,256])
    matrix_recurrent[index_w_1] = 0
    
    nsetup.mapper._program_onchip_programmable_connections(matrix_recurrent)
    nsetup.mapper._program_onchip_broadcast_programmable(matrix_b)
    nsetup.mapper._program_onchip_weight_matrix_programmable(matrix_weight) #broadcast goes to weight 1 the rest is w 2
    nsetup.mapper._program_onchip_exc_inh(matrix_e_i)
    
    #set neuron parameters
    #chip.configurator.set_parameter("IF_TAU2_N",3.3e-9)
    #chip.configurator.set_parameter("IF_DC_P",23.9e-11)
    #chip.configurator.set_parameter("VA_EXC_N",2.3e-5)
    #chip.configurator.set_parameter("VDPIE_TAU_P",82.0e-12)
    #chip.configurator.set_parameter("VDPIE_THR_P",82.0e-12)
    #chip.configurator.set_parameter("IF_THR_N",1000.0e-12)
    
    #chec if the neuron can get excited...
    #index_neu_zero_up = inputpop.synapses['virtual_exc'].addr['neu'] == 244
    #syn = inputpop.synapses['virtual_exc'][index_neu_zero_up]
    #spktrain = syn.spiketrains_regular(100)
    #nsetup.stimulate(spktrain,send_reset_event=False)
    #index_neu_zero_up = inputpop.synapses['programmable_exc'].addr['neu'] == 244
    #syn = inputpop.synapses['virtual_exc'][index_neu_zero_up]
    #spktrain = syn.spiketrains_regular(100)
    #nsetup.stimulate(spktrain,send_reset_event=False)

    #map bioamp to neuromorphic chip
    #bioamp.map_bioamp_delta_single_dest(210)
    if(one_neuron_simple_exp == True):
        bioamp.map_bioamp_onetomany(210,211)
        duration_exp = 1000    
        out = nsetup.stimulate({},duration=duration_exp)
            
        raw_data_out = out[0].raw_data() 
        features_neu = [210,211]
        for this_neu in range(2):
            neuron_index = np.where(raw_data_out[:,1] == features_neu[this_neu])   
            if(len(neuron_index[0]) > 2): 
                plot(np.linspace(0,duration_exp,len(np.diff(raw_data_out[neuron_index,0])[0])), np.diff(raw_data_out[neuron_index,0])[0], 'o-')
                
        np.savetxt('bioamp/simple_receptive_neuron_spikes_times.txt', raw_data_out[:,0]) 
        np.savetxt('bioamp/simple_receptive_neuron_spikes_ids.txt', raw_data_out[:,1])        
  

    if(population_coding == True):
        neudest_up = np.random.choice(np.linspace(0,125,126),6)
        neudest_dn = np.random.choice(np.linspace(126,255,130),6)
        bioamp._init_fpga_mapper()
        #bioamp.map_bioamp_reservoir(neudest_up,neudest_dn)
        bioamp.map_bioamp_reservoir_broadcast(n_columns=3)
        nsetup.mapper._program_detail_mapping(0)    
          
        duration_rec = 350
        #out = nsetup.stimulate({},duration=duration_rec)

        
        #init liquid state machine
        #populate neurons
        #rcnpop = pyNCS.Population('neurons', 'for fun') 
        #rcnpop.populate_all(nsetup,'mn256r1','excitatory')

        #liquid = L.Lsm(rcnpop)
        nsetup.mapper._program_detail_mapping(2**3+2**4)  
        out = nsetup.stimulate({},duration=duration_rec)
        nsetup.mapper._program_detail_mapping(0)   


        def go_reconstruct_signal(duration_rec,upch=300,dnch=305,delta_up=0.1,delta_dn=0.0725, do_plot=True, do_detrend=False):
                    
            out = nsetup.stimulate({},duration=duration_rec)
            raw_data = out[0].raw_data()
            index_dn = np.where(raw_data[:,1] == dnch)[0]
            index_up = np.where(raw_data[:,1] == upch)[0]

            raw_data_input = []
            raw_data_input.extend(raw_data[index_dn,:])
            raw_data_input.extend(raw_data[index_up,:])
            raw_data_input = np.reshape(raw_data_input,[len(index_dn)+len(index_up),2])

            raw_data = np.delete(raw_data, index_dn,axis=0)
            raw_data = np.delete(raw_data, index_up,axis=0)

            index_sort = np.argsort(raw_data_input[:,0])   
            #reconstruct from up and down
            up_index = np.where(raw_data_input[:,1]==upch)
            dn_index = np.where(raw_data_input[:,1]==dnch)
            index_ord =  np.argsort(raw_data_input[:,0])   
            signal = np.zeros([len(raw_data_input)])
            for i in range(1,len(raw_data_input)):
                if(raw_data_input[index_ord[i],1] == upch):
                    signal[i] = signal[i-1] + delta_up
                if(raw_data_input[index_ord[i],1] == dnch):
                    signal[i] = signal[i-1] - delta_dn
                    
            signal_trace_good = [raw_data_input[index_ord,0],signal]
            if do_detrend:
                df = scipy.signal.detrend(signal_trace_good[1])
                signal_trace_good  = np.array([signal_trace_good[0],df]).transpose()
            else:
                signal_trace_good  = np.array([signal_trace_good[0],signal_trace_good[1]]).transpose()  
                
            if do_plot == True:
                figure()
                plot(signal_trace_good[:,0],signal_trace_good[:,1], 'o-')
                
            return signal_trace_good
            
        signal = go_reconstruct_signal(3000,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)


        def encode_and_teach(duration_rec=6000,delta_mem=50):
        
            #chip.load_parameters('biases/biases_reservoir_synthetic_stimuli.biases')
            #then load bioamp biases on top of these
            rcnpop = pyNCS.Population('neurons', 'for fun') 
            rcnpop.populate_all(nsetup,'mn256r1','excitatory')
            res = L.Reservoir(rcnpop, cee=1.0, cii=0.45)
            res.program_config()    
            #c = 0.2
            #dim = np.round(np.sqrt(len(liquid.rcn.synapses['virtual_exc'].addr)*c))
                
            #reset previous learning
            res.reset()
            
            figs = figure()
            # Time vector for analog signals
            Fs    = 1000/1e3 # Sampling frequency (in kHz)
            T     = duration_rec
            nT    = np.round (Fs*T)
            timev = np.linspace(0,T,nT)
            #Conversion from spikes to analog
            membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
            out = nsetup.stimulate({},duration=duration_rec)
            signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)
        
            #extract input and output
            raw_data = out[0].raw_data()
            dnch = 300
            upch = 305
            index_dn = np.where(raw_data[:,1] == dnch)[0]
            index_up = np.where(raw_data[:,1] == upch)[0]
            raw_data_input = []
            raw_data_input.extend(raw_data[index_dn,:])
            raw_data_input.extend(raw_data[index_up,:])
            raw_data_input = np.reshape(raw_data_input,[len(index_dn)+len(index_up),2])
            index_up = np.where(raw_data_input[:,1] == upch)[0]
            index_dn = np.where(raw_data_input[:,1] == dnch)[0]
            raw_data_input[index_dn,1] = 1
            raw_data_input[index_up,1] = 0   
            raw_data = out[0].raw_data()
            index_to_del = np.where(raw_data[:,1] > 255)
            raw_data = np.delete(raw_data, index_to_del,axis=0)
            
            #teach on reconstructed signal
            X = L.ts2sig(timev, membrane, raw_data_input[:,0], raw_data_input[:,1], n_neu = 256)
            Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)     
            #teaching signal is interpolated to match actual len of signals
            
            signal_ad = [signal[:,0], signal[:,1]]
            ynew = np.linspace(np.min(signal_ad[0]),np.max(signal_ad[0]),nT+1)
            s = interpolate.interp1d(signal_ad[0], signal_ad[1],kind="cubic")
            teach_sig = s(ynew)  
            
            res.train(X,Y,teach_sig[0:len(X),None])
            zh = res.predict(X, Y)
            
            figure()
            subplot(3,1,1)
            plot(timev[0::],teach_sig[1::],label='teach signal')
            legend(loc='best')
            subplot(3,1,2)
            plot(timev[0::],zh["input"], label='input')
            legend(loc='best')
            subplot(3,1,3)
            plot(timev[0::],zh["output"], label='output')
            legend(loc='best')
                
            figure()        
            index_non_zeros = []
            for i in range(256):
                if(np.sum(Y[:,i]) != 0):
                    index_non_zeros.append(i)  
            size_f = np.floor(np.sqrt(len(index_non_zeros)))
            for i in range(int(size_f**2)):
                #subplot(size_f,size_f,i) 
                plot(Y[:,index_non_zeros[i]])  
                #axis('off')  
                

        

        def mini_encoding(duration_rec = 3000, delta_mem = 10):
            #from scipy.fftpack import fft
            #yyf = fft(Y[:,index_non_zeros[i]])

            import wave
            import sys
            spf = wave.open('/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/Hackenelektrodenableitung_mecopoda_elongata_chirper_2_trim.wav','r')
            #Extract Raw Audio from Wav File
            signal = spf.readframes(-1)
            signal = np.fromstring(signal, 'Int16')
            fs = spf.getframerate()
            time_orig=np.linspace(0, len(signal)/fs, num=len(signal))
            
            
            from scipy import interpolate
            s = interpolate.interp1d(time_orig*1000, signal,kind="linear")
            time_n  = np.linspace(0,np.max(time_orig)*1000,10000)
            ynew = s(time_n)#interpolate.splev(xnew, tck, der=0)
            
            
            #figure()
            #plot(time_orig,signal)
            mul_f = np.ceil(duration_rec/(np.max(time_orig)*1000))
            time_orig = np.linspace(0,np.max(time_n)*mul_f, len(time_n)*mul_f)
            signal_f = []
            for i in range(int(mul_f)):
                signal_f.append(ynew)
            signal = np.reshape(signal_f,[len(ynew)*mul_f])

            figure()
            plot(time_orig,signal)


            #fig_h = figure()
            fig_hh = figure()
        
            # Time vector for analog signals
            Fs    = 100/1e3 # Sampling frequency (in kHz)
            T     = duration_rec
            nT    = np.round (Fs*T)
            timev = np.linspace(0,T,nT)

            #Conversion from spikes to analog
            membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
         
            out = nsetup.stimulate({},duration=duration_rec)
            raw_data = out[0].raw_data()
            dnch = 300
            upch = 305
            index_dn = np.where(raw_data[:,1] == dnch)[0]
            index_up = np.where(raw_data[:,1] == upch)[0]

            raw_data_input = []
            raw_data_input.extend(raw_data[index_dn,:])
            raw_data_input.extend(raw_data[index_up,:])
            raw_data_input = np.reshape(raw_data_input,[len(index_dn)+len(index_up),2])
            index_up = np.where(raw_data_input[:,1] == upch)[0]
            index_dn = np.where(raw_data_input[:,1] == dnch)[0]
            raw_data_input[index_dn,1] = 1
            raw_data_input[index_up,1] = 0
            X = L.ts2sig(timev, membrane, raw_data_input[:,0], raw_data_input[:,1], n_neu = 4)       
            figure()
            for i in range(2):
                plot(X[:,i])
            
            raw_data = out[0].raw_data()
            index_to_del = np.where(raw_data[:,1] > 255)
            raw_data = np.delete(raw_data, index_to_del,axis=0)
            Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)
        
            index_non_zeros = []
            for i in range(256):
                if(np.sum(Y[:,i]) != 0):
                    index_non_zeros.append(i)  
            size_f = np.floor(np.sqrt(len(index_non_zeros)))
            figure(fig_hh.number)
            for i in range(int(size_f**2)):
                #subplot(size_f,size_f,i) 
                plot(Y[:,index_non_zeros[i]])  
                #axis('off')  
            figure()
            for i in range(int(size_f**2)):
                subplot(size_f,size_f,i) 
                plot(Y[:,index_non_zeros[i]])  
                ylim([0,3])
                axis('off')      
                
            figure()
            #raster plot lsm   
            plot(raw_data[:,0], raw_data[:,1], '*', markersize=2)
            ylim([0,256])
            xlim([0,duration_rec])
            xlabel('Time [ms]')
            ylabel('Neu Id')
            
  
        def go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.0725, do_plot=True, do_detrend=False):
                    
            #out = nsetup.stimulate({},duration=duration_rec)
            raw_data = out[0].raw_data()
            index_dn = np.where(raw_data[:,1] == dnch)[0]
            index_up = np.where(raw_data[:,1] == upch)[0]

            raw_data_input = []
            raw_data_input.extend(raw_data[index_dn,:])
            raw_data_input.extend(raw_data[index_up,:])
            raw_data_input = np.reshape(raw_data_input,[len(index_dn)+len(index_up),2])

            raw_data = np.delete(raw_data, index_dn,axis=0)
            raw_data = np.delete(raw_data, index_up,axis=0)

            index_sort = np.argsort(raw_data_input[:,0])   
            #reconstruct from up and down
            up_index = np.where(raw_data_input[:,1]==upch)
            dn_index = np.where(raw_data_input[:,1]==dnch)
            index_ord =  np.argsort(raw_data_input[:,0])   
            signal = np.zeros([len(raw_data_input)])
            for i in range(1,len(raw_data_input)):
                if(raw_data_input[index_ord[i],1] == upch):
                    signal[i] = signal[i-1] + delta_up
                if(raw_data_input[index_ord[i],1] == dnch):
                    signal[i] = signal[i-1] - delta_dn
                    
            signal_trace_good = [raw_data_input[index_ord,0],signal]
            if do_detrend:
                df = scipy.signal.detrend(signal_trace_good[1])
                signal_trace_good  = np.array([signal_trace_good[0],df]).transpose()
            else:
                signal_trace_good  = np.array([signal_trace_good[0],signal_trace_good[1]]).transpose()  
                
            if do_plot == True:
                figure(figs.number)
                plot(signal_trace_good[:,0],signal_trace_good[:,1])
                
            return signal_trace_good

        def test_determinism(duration_rec, n_trial=3, delta_mem=60, delta_up=15, plot_svd = False):
            fig_h = figure()
            fig_hh = figure()
            figs = figure()
            # Time vector for analog signals
            Fs    = 100/1e3 # Sampling frequency (in kHz)
            T     = duration_rec
            nT    = np.round (Fs*T)
            timev = np.linspace(0,T,nT)

            #Conversion from spikes to analog
            membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
            membrane_up = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_up**2)))
            
            #nsetup.mapper._program_detail_mapping(2**3+2**4)  
            command1 = subprocess.Popen(['sh', '/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/do_stim.sh'])
            out = nsetup.stimulate({},duration=duration_rec)
            command1 = subprocess.Popen(['killall', 'aplay'])
            all_sign = []
            for this_trial in range(n_trial):
                command1 = subprocess.Popen(['sh', '/home/federico/project/work/trunk/data/Insects/Insect Neurophys Data/do_stim.sh'])
                #time.sleep(0.2)
                out = nsetup.stimulate({},duration=duration_rec)
                signal = go_reconstruct_signal_from_out(out,figs,upch=300,dnch=305,delta_up=0.1,delta_dn=0.1,do_detrend=False)
                all_sign.append(signal)
                command1 = subprocess.Popen(['killall', 'aplay'])
                #time.sleep(1)
                #nsetup.mapper._program_detail_mapping(0) 
                raw_data = out[0].raw_data()
                index_to_del = np.where(raw_data[:,1] > 255)
                raw_data = np.delete(raw_data, index_to_del,axis=0)
                Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)
                figure(fig_h.number)
                for i in range(255):
                    subplot(16,16,i)
                    plot(Y[:,i])
                    axis('off')
              
                if (this_trial == 0):
                    index_non_zeros = []
                    for i in range(256):
                        if(np.sum(Y[:,i]) != 0):
                            index_non_zeros.append(i)  
                    size_f = np.floor(np.sqrt(len(index_non_zeros)))
                figure(fig_hh.number)
                for i in range(int(size_f**2)):
                    subplot(size_f,size_f,i) 
                    plot(Y[:,index_non_zeros[i]])  
                    axis('off')
            
            #out = nsetup.stimulate({},duration=duration_rec)  
            
              
            if(plot_svd == True): 
                #SVD
                figure()
                ac=np.mean(Y**2,axis=0)
                #aci=np.mean(X**2,axis=0)
                max_pos = np.where(ac == np.max(ac))[0]
                #max_posi = np.where(aci == np.max(aci))[0]
                subplot(3,1,1)
                plot(signal[:,0],signal[:,1])
                subplot(3,1,2)
                plot(Y[:,max_pos])
                subplot(3,1,3)
                CO = np.dot(Y.T,Y)
                CI = np.dot(signal,signal.T)
                si = np.linalg.svd(CI, full_matrices=False, compute_uv=False)
                so = np.linalg.svd(CO, full_matrices=True, compute_uv=False)
                semilogy(so/so[0], 'bo-', label="outputs")
                semilogy(si/si[0], 'go-', label="inputs")
                legend(loc="best")   
            
            raw_data = out[0].raw_data()
            index_to_del = np.where(raw_data[:,1] > 255)
            raw_data = np.delete(raw_data, index_to_del,axis=0)
            membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
            Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)
            
            figure()
            #raster plot lsm   
            vlines(raw_data[:,0], raw_data[:,1] + .5, raw_data[:,1] + 1.5)
            ylim([0,256])
            xlim([0,duration_rec])
            xlabel('Time [ms]')
            ylabel('Neu Id')
  
            figure()
            #raster plot lsm   
            plot(raw_data[:,0], raw_data[:,1], '*', markersize=2)
            ylim([0,256])
            xlim([0,duration_rec])
            xlabel('Time [ms]')
            ylabel('Neu Id')
        
        #number or recordings
        def go_plot_reservoir(duration_rec, delta_mem=50, delta_up = delta_mem/4.0):
            # Time vector for analog signals
            Fs    = 100/1e3 # Sampling frequency (in kHz)
            T     = duration_rec
            nT    = np.round (Fs*T)
            timev = np.linspace(0,T,nT)

            #Conversion from spikes to analog
            membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_mem**2)))
            membrane_up = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*delta_up**2)))
            
            #nsetup.mapper._program_detail_mapping(2**3+2**4)  
            time.sleep(2)
            out = nsetup.stimulate({},duration=duration_rec)
            #nsetup.mapper._program_detail_mapping(0) 
            raw_data = out[0].raw_data()
            index_to_del = np.where(raw_data[:,1] > 255)
            raw_data = np.delete(raw_data, index_to_del,axis=0)
            Y = L.ts2sig(timev, membrane, raw_data[:,0], raw_data[:,1], n_neu = 256)
            figure()
            for i in range(255):
                subplot(16,16,i)
                plot(Y[:,i])
                axis('off')
             
            upch = 300
            dnch = 305
            deltaup = 0.1
            deltadn = 0.0725
            raw_data = out[0].raw_data()
            index_dn = np.where(raw_data[:,1] == dnch)[0]
            index_up = np.where(raw_data[:,1] == upch)[0]

            raw_data_input = []
            raw_data_input.extend(raw_data[index_dn,:])
            raw_data_input.extend(raw_data[index_up,:])
            raw_data_input = np.reshape(raw_data_input,[len(index_dn)+len(index_up),2])

            raw_data = np.delete(raw_data, index_dn,axis=0)
            raw_data = np.delete(raw_data, index_up,axis=0)
            index_sort = np.argsort(raw_data_input[:,0])   
            
            #reconstruct from up and down
            up_index = np.where(raw_data_input[:,1]==upch)
            dn_index = np.where(raw_data_input[:,1]==dnch)
            index_ord =  np.argsort(raw_data_input[:,0])   
            signal = np.zeros([len(raw_data_input)])
            for i in range(1,len(raw_data_input)):
                if(raw_data_input[index_ord[i],1] == upch):
                    signal[i] = signal[i-1] + deltaup
                if(raw_data_input[index_ord[i],1] == dnch):
                    signal[i] = signal[i-1] - deltadn
            figure()        
            plot(raw_data_input[index_ord,0],signal+45)
            signal_trace_good = [raw_data_input[index_ord,0],signal]
            
            raw_data_input[up_index,1] = 0
            raw_data_input[dn_index,1] = 1
            X = L.ts2sig(timev, membrane_up, raw_data_input[:,0], raw_data_input[:,1], n_neu = 2)
            figure()    
            for i in range(2):
                subplot(2,1,i)
                plot(X[:,i])
                axis('off')            

            #SVD
            figure()
            ac=np.mean(Y**2,axis=0)
            aci=np.mean(X**2,axis=0)
            max_pos = np.where(ac == np.max(ac))[0]
            max_posi = np.where(aci == np.max(aci))[0]
            subplot(3,1,1)
            plot(X[:,max_posi])
            subplot(3,1,2)
            plot(Y[:,max_pos])
            subplot(3,1,3)
            CO = np.dot(Y.T,Y)
            CI = np.dot(X.T,X)
            si = np.linalg.svd(CI, full_matrices=True, compute_uv=False)
            so = np.linalg.svd(CO, full_matrices=True, compute_uv=False)
            semilogy(so/so[0], 'bo-', label="outputs")
            semilogy(si/si[0], 'go-', label="inputs")
            legend(loc="best")    
      
            figure()
            #raster plot lsm   
            vlines(raw_data[:,0], raw_data[:,1] + .5, raw_data[:,1] + 1.5)
            ylim([0,256])
            xlim([0,duration_rec])
            xlabel('Time [ms]')
            ylabel('Neu Id')
        

        
        bins = 350
        nbins = duration_rec / bins
        mean_rates = []
        for i in range(nbins):
            mean_rates_tmp = functions.meanNeuFiring(raw_data,np.linspace(0,255,256),[i*bins,(i+1)*bins])
            mean_rates.append(mean_rates_tmp)
            
        figure()
        hold('on')    
        neu_to_plot = np.linspace(0,255,256).astype('int')
        for i in range(nbins):
            plot(np.repeat(i,len(neu_to_plot)),mean_rates[i][neu_to_plot])
        
        #record ad
        duration_exp = 1000    


        out = nsetup.stimulate({},duration=10000)
        deltaup = 0.1
        deltadn = 0.081
        upch = 300
        dnch = 305
        #reconstruct from up and down
        raw_data = out[0].raw_data() 
        up_index = np.where(raw_data[:,1]==upch)
        dn_index = np.where(raw_data[:,1]==dnch)
        index_ord =  np.argsort(raw_data[:,0])   
        signal = np.zeros([len(raw_data)])
        for i in range(1,len(raw_data)):
            if(raw_data[index_ord[i],1] == upch):
                signal[i] = signal[i-1] + deltaup
            if(raw_data[index_ord[i],1] == dnch):
                signal[i] = signal[i-1] - deltadn
        plot(raw_data[index_ord,0],signal+45)
        signal_trace_good = [raw_data[index_ord,0],signal]

        xlim([1000,1280])
        ylim([0,20])
        vlines(raw_data[up_index,0][0] , .5, 1.0, color ='g' )
        vlines(raw_data[dn_index,0][0] , 1.5, 2.0, color='r')
        #np.savetxt("bioamp/raw_data_ad_tiime.txt",raw_data[index_ord,0])
        #np.savetxt("bioamp/raw_data_ad_amp.txt",signal)
    
        
    #features neu
    if(perceptrons_experiment == True):
        ## FEATURES
        features_neu = np.linspace(0,10,11)       
        features_pop = pyNCS.Population('','')
        features_pop.populate_by_id(nsetup,'mn256r1','excitatory', features_neu) 
        ## PERCEPTRONS
        perceptrons_neu = np.linspace(128,255,128)       
        perceptrons_pop = pyNCS.Population('','')
        perceptrons_pop.populate_by_id(nsetup, 'mn256r1','excitatory', perceptrons_neu) 
        
        matrix_e_i, w_programmable, tot_up_post_addr, tot_dn_post_addr = bioamp.map_bioamp_features_neu(features_neu, n_syn_per_neu = 32, syntype='programmable', weight=3)
        
        duration_exp = 1000    
        out = nsetup.stimulate({},duration=duration_exp)
            
        raw_data_out = out[0].raw_data() 
        for this_neu in range(len(features_neu)):
            neuron_index = np.where(raw_data_out[:,1] == features_neu[this_neu])   
            if(len(neuron_index[0]) > 2): 
                plot(np.linspace(0,duration_exp,len(np.diff(raw_data_out[neuron_index,0])[0])), np.diff(raw_data_out[neuron_index,0])[0], 'o-')
         
        #randomly connect neuron in the network.. reservoirizziamo!
        connections = np.random.choice([0,1],[256,256])
        weights = np.random.choice([0,1,2,3],[256,256])
        exc_inh = np.random.choice([0,1],[256,256])
        #nsetup.mapper._program_onchip_programmable_connections(connections)
        #nsetup.mapper._program_onchip_weight_matrix_programmable(weights)
        #nsetup.mapper._program_onchip_exc_inh(exc_inh)
        
        bioamp.connect_perceptrons_with_features(perceptrons_pop, features_pop,  cee=1.0, n_class = 2)
        
  
if(ckeck_variability_input == True):    
    
    out = nsetup.stimulate({},duration = 1000)
    spk_out = out[0]
    
if(scope_capture == True):
    import pyAgilent
    from pylab import *
    import matplotlib

    #init oscilloscope
    osc = pyAgilent.Agilent(host="172.19.10.159");
    osc._send_command('WAV:FORM asc');

    membrane = osc._read_data_from_channel(1)
    inputamp = osc._read_data_from_channel(4)
    outputamp = osc._read_data_from_channel(3)
    
    time = np.linspace(0,1000,len(membrane))
    time_in = np.linspace(0,1000,len(inputamp))
    time_out = np.linspace(0,1000,len(outputamp))   
    
    figure()
    plot(time,membrane)
    plot(time_in,inputamp)
    plot(time_out,outputamp)
    xlabel('time [ms]')
    ylabel('amplitude [V]')
    
    np.savetxt("bioamp/membrane_4.txt", membrane)
    np.savetxt("bioamp/membrane_time_4.txt", time)
    np.savetxt("bioamp/inputamp_4.txt", inputamp)
    np.savetxt("bioamp/inputamp_time_4.txt", time_in)
    np.savetxt("bioamp/outputamp_4.txt", outputamp)
    np.savetxt("bioamp/outputamp_time_4.txt", time_out)
    
        
    
    
    
    
    
    
