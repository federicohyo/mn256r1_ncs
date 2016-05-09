import threading
import Queue
import socket
#For probing network buffer
import fcntl
from termios import FIONREAD
from contextlib import contextmanager
#AER related
import pyNCS.pyST as pyST
from aexutils import *

#Special events used to control reset and synchonization
GOEVENT = np.array([-1, 0], dtype='uint32').tostring() #for stim = None, this ensures that the hold is released.
HOLDEVENT = np.array([-2, 0], dtype='uint32').tostring() #Hold and reset: the aex device file is opened immediately before the next event is receied



class AEXClientBase(threading.Thread):
    '''
    This class starts a Monitor client for the AEX server. Once created, this continously reads the monitored events.
    '''
    #Singleton: make sure a client is only instantiated once in a session
    def __init__(self,
            host='localhost',
            MonChannelAddress=None,
            channels=None,
            fps=25.,
            port=50001,
            port_control=50003,
            autostart=True,
            qsize=2048,
            ):
        '''
        *MonChannelAddress:* Monitor Channel Addressing object, if omitted, default channel is taken
        *channels:* Monitor these channels instead (not tested yet)
        *fps:* Number of fetches per second. Corresponds to the inverse of the monitor client socket timeout. if a real-time visualization software is used, this must be equal to its frame rate. (Default 25.)
        *host:* Monitoring AEX server hostname.
        *port:* Port of the Monitoring Server (Default 50001)
        *qsize:* The maxsize of the queue
        '''
        threading.Thread.__init__(self)
        self.daemon = True  # The entire Python program exits when no alive non-daemon threads are left
        print "Connecting to " + host
        #self.aexfd = os.open("/dev/aerfx2_0", os.O_RDWR | os.O_NONBLOCK)
        self.finished = threading.Event()
        self.buffer = Queue.Queue(qsize)  # 8192 packets in Queue=~5min

        if MonChannelAddress == None:
            self.MonChannelAddress = pyST.getDefaultMonChannelAddress()
        else:
            self.MonChannelAddress = MonChannelAddress

        self.recvlock = threading.Lock()
        self._nbufferempty = False
        self.eventsPacket = np.array([])
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port_mon = port
        self.port_control = port_control

        self.sock.connect((self.host, self.port_mon))
        # Set the socket to non-blocking (with timeout) to avoid the thread
        # being stuck. Because of this, I must catch EWOULDBLOCK error
        #Frame period
        self.fT = 1. / float(fps)
        self.sock.settimeout(self.fT)
        if channels == None:
            #get number of channels
            self.channels = range(len(self.MonChannelAddress))
        else:
            self.channels = channels

        #Build decoder functions
        if autostart:
            self.start()

    def mask_monitor(self, mask, check_value):
        sock_control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_control.connect((self.host, self.port_control))
        sock_control.sendall('addMask(%s, %d, %d);' % (self.sock.
            getsockname(), mask, check_value))
        sock_control.close()

    def run(self):
        while self.finished.isSet() != True:
            t0 = time.time()

            with self.recvlock:
                self.fetch_raw()

            #Fastest frame rate should be fT. Wait in case we've been faster
            t_left = self.fT - (time.time() - t0)
            if t_left > 0:
                time.sleep(t_left)
            # Ok now check how much data is available and choose a multiple of
            # 8 (AE packet size

    def flush(self, verbose=False):
        '''
        empties the buffers.
        '''

        #empty Queue buffer
        with self.recvlock:
            while True:
                try:
                    self.buffer.get(False)
                except Queue.Empty:
                    break
            while True:
                try:
                    tmp = self.sock.recv(1024)
                    if len(tmp) < 1024:
                        break  # avoid endless loop
                    if verbose:
                        print 'flush: received {0} bytes'.format(len(tmp))
                except socket.timeout:
                    break
                except socket.error, err:
                    if err.errno != errno.EWOULDBLOCK:
                        raise err
                    break

    def stop(self):
        self.finished.set()
        with self.recvlock:
            self.sock.close()

    def __del__(self):
        self.stop()


##################
##Mapper functions
class AEXMonClient(AEXClientBase):

    def _recv_with_check(self):
        buf_n = np.fromstring(
            fcntl.ioctl(self.sock, FIONREAD, "xxxx"), 'uint32')
        recv_n = 8 * (buf_n / 8)
        if not buf_n < 8:
            try:
                ev = self.sock.recv(recv_n)
                nev = len(ev) / 8
            except socket.timeout:
                nev = 0
                pass
            except socket.error, err:
                if err.errno != errno.EWOULDBLOCK:
                    raise err
        else:
            return '', -1

        return ev, nev

    def put_buffer(self, ev):
        try:
            self.buffer.put(ev, block=False)
        #Drop last and put new
        except Queue.Full:
            self.buffer.get(block=True)
            self.buffer.put(ev, block=True)

    @contextmanager
    def _isbuffernotempty(self, ev, nev):
        try:
            yield
            self._nbufferempty = True
        except ValueError:
            if not ev == '' and nev == -1:
                self._nbufferempty = True  # queue is not empty
                pass
            else:
                self._nbufferrempty = False  # queue is not empty
                pass

    def _extract_events(self, ev):
        if ev == '':
            raise ValueError()

        tmp_array = np.fromstring(ev, dtype='uint32')

        #Do swap if big endian system
        if sys.byteorder == 'big':
            tmp_array.byteswap(True)
        evs = events(np.fliplr(tmp_array.reshape(-1, 2)), atype='p')

        return evs

    def fetch_raw(self):
        '''
        This functions is called by threading.Thread.run to read from the TCP fifo and add an event packet (a STas.events object) to the client's buffer in physical format.
        '''
        eventsPacket, nev = self._recv_with_check()
        
        with self._isbuffernotempty(eventsPacket, nev):
            #Read from binary
            evs = self._extract_events(eventsPacket)
            #extract logical for all channels
            self.put_buffer(evs)

        return self._nbufferempty

    def fetch(self):
        '''
        Empties the buffer and returns its contents
        returns numpy.array in raw address - timestamp format.
        '''
        x = events(atype='p')
        while True:
            try:
                x.add_adtmev(self.buffer.get(block=False).get_adtmev())
            except Queue.Empty:
                return x.get_adtmev()

    def listen(self, tDuration=None, normalize=True, filter_duplicates=False):
        '''
        empties the queue & returns a SpikeList containing the data that was in the Queue
        synposis: client.flush(); time.sleep(1); out=client.listen()
        @author emre@ini.phys.ethz.ch, andstein@student.ethz.ch

        *tDuration*: is a float defining the duration to be listened (in ms)
        *output*: is a string defining in what form the output should be returned. Default is SpikeList where a NeuroTools.signal SpikeList is returned. Other possiblities are 'array' where a normalized numpy array is returned and 'raw' where a pyST.channelEvents object is returned.
        *normalize*: if output is 'SpikeList' then normalize defines whether the output should be normalized using pyST.normalizeAER
        *filter_duplicates* if True erase in all channels double events within the a 0.01ms time-frame (buggy hw)
        '''
        n = self.buffer.qsize()

        #initialize output channelEvents
        out = events(atype='p')

        while n > 0:
            try:
                evs = self.buffer.get(block=True, timeout=10e-3)
                n -= 1
            except Queue.Empty:
                break

            if evs.get_nev() > 0:
                t0 = time.time()
                out.add_adtm(evs.get_ad(), evs.get_tm())
            #Assumes isi
            if tDuration != None and out.get_nev() > 0:
                if (out.get_tm()[-1] - out.get_tm()[0]) > tDuration * 1000:
                    break

        return out.get_adtmev()


class AEXClient(AEXMonClient):
    '''
    This class starts a Monitor/Stimulation client for the AEX server. Once started, this continously reads the monitored events.
    '''
    def __init__(self,
            host='localhost',
            host_stim=None,
            MonChannelAddress=None,
            SeqChannelAddress=None,
            channels=None,
            fps=25.,
            port_mon=50001,
            port_stim=50002,
            port_control=50003,
            autostart=True,
            qsize=2048,
            ):
        '''
        *MonChannelAddress:* Monitor Channel Addressing object, if omitted, default channel is taken
        *SeqChannelAddress:* Sequencer Channel Addressing object, if omitted, default channel is taken
        *channels:* Monitor these channels instead (not tested yet)
        *fps:* Number of fetches per second. Corresponds to the inverse of the monitor client socket timeout. if a real-time visualization software is used, this must be equal to its frame rate. (Default 25.)
        *host:* Monitoring and Sequencing AEX server hostname. (Must be same)
        *port_mon:* Port of the Monitoring Server (Default 50001)
        *port_stim:* Port of the Monitoring Server (Default 50002)
        *qsize*: is the size of the queue (FIFO). This is automatically adjusted if the buffer is too small.
        '''

        if MonChannelAddress == None:
            self.MonChannelAddress = pyST.getDefaultMonChannelAddress()
        else:
            self.MonChannelAddress = MonChannelAddress
        if SeqChannelAddress == None:
            self.SeqChannelAddress = pyST.getDefaultSeqChannelAddress()
        else:
            self.SeqChannelAddress = SeqChannelAddress

        AEXMonClient.__init__(
                self, MonChannelAddress=self.MonChannelAddress,
                channels=channels,
                fps=fps, host=host,
                port=port_mon,
                port_control=port_control,
                autostart=False,
                qsize=qsize,
                )

        if host_stim == None:
            self.host_stim = host
        else:
            self.host_stim = host_stim

        #self.aexfd = os.open("/dev/aerfx2_0", os.O_RDWR | os.O_NONBLOCK)
        self.stim_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.stim_sock.connect((self.host_stim, port_stim))

        if autostart:
            self.start()

    def stimulate(self,
            stim=None,
            tDuration=None,
            context=None,
            filter_duplicates=False,
            send_reset_event=True,
            debug=False,
            verbose=False,
            ):
        '''
        NOTE: OLD DOCUMENTATION !!
        This function stimulates through AEX server. **NOT** network lag invariant!
        *stim:* should be a SpikeList (usable by channelAddressing.exportAER())
        *tDuration:* in ms (Default: guess from stim or 1000 if stim==None).
        *isi:* Whether the input should be send to server as inter-spike intervals.
        *send_reset_event* resets the AEX board when stimulated (experimental).
        *output* format of output, spikelist/array/raw/
        *context* is a context which is executed before and after the stimulation (for sync functions and clean-up functions for example, see the python contextlib module).
        *filter_duplicates* if True erase in all channels double events within the a 0.01ms time-frame (buggy hw)
        '''

        if context == None:
            context = empty_context

        stim = events(stim, 'p')

        if sys.byteorder == 'big':
            stimByteStream = stim.get_tmadev().byteswap().tostring()
        else:
            stimByteStream = stim.get_tmadev().tostring()

        if verbose:
            print "Bytecoding input... done"
            print np.fromstring(stimByteStream, 'uint32')

        if tDuration == None:
            tDuration = np.sum(stim.get_tm()) * 1e-3  # from us to ms

        if self.buffer.maxsize / self.fT < tDuration:
            while self.buffer.maxsize / self.fT < tDuration:
                self.buffer.maxsize *= 2

        #Clean up pre-stimulus data
        if verbose:
            print "Flushing pre-stimulus data"
        self.flush()

        if verbose:
            print "Sending to " + self.host

        if len(stim)>0:
            if verbose:
                print np.fromstring(stimByteStream, 'uint32')

            if send_reset_event:
                self.stim_sock.send(HOLDEVENT)
                self.flush(
                    )  # Be sure that there are no events left in the buffer
                # Since the monitor is held, no events can come in until the
                # reset. Well... in theory
                #Bug: Somehow, a few event still get through...
            with context():
                self.buffer.put(-1, block=True)
                self.stim_sock.send(stimByteStream)
                time.sleep(tDuration * 1e-3 + tDuration * 0.3 * 1e-3 + .100)

        else:
            if verbose:
                print "Waiting " + str(
                    tDuration) + "ms " + "for stimulation to finish"
            if send_reset_event:
                self.stim_sock.send(HOLDEVENT)
                self.flush()

            with context():
                self.buffer.put(-1, block=True)
                self.stim_sock.send(GOEVENT)
                time.sleep(tDuration * 1e-3 + tDuration * 0.3 * 1e-3 + .100)

        #####
        # Throw away pre-stimulus data
        if verbose:
            print "Throwing away pre-stimulus-data"

        while self.buffer.get() != -1:
            pass

        if send_reset_event:
            normalize = False
        else:
            normalize = True

        if not debug:  # useful for debugging
            return self.listen(tDuration=tDuration,
                               normalize=normalize,
                               filter_duplicates=filter_duplicates)
            
    def stop(self):
        self.finished.set()
        with self.recvlock:  # Be sure that there is no fetching going on
            self.sock.close()
        #with self.stimlock # there is no stimlock
        self.stim_sock.close()

    def __del__(self):
        self.stop()

            
class AEXRelayBuffer(AEXClient):
    def __init__(self, mapping=None,**kwargs):
        """
        A "relay" class for monitoring events in a setup and stimulating directly, given a mapping_table. This class acts like a 'software mapper'.
        The mapping table is a list should be a list acceptable by pyAex.setMappings (taking the mapper library version into account!)
        Additional keyword arguments are passed to pyAex.AEXclient.
        This thread appends to a queue but does not stimulate. For stimulation, it must be used in conjunction with AEXRelay.
        """
        if mapping is not None:
            #make a lookup table out of the dictionary (its faster)
            self.mapping_dict = dlist_to_dict(mapping)
        else:
            self.mapping_dict = None

        AEXClient.__init__(self, **kwargs)

    def fetch_and_relay(self):
        ev_str, nev = self._recv_with_check()
        if nev > 0: 
            #Read from binary
            evs = self._extract_events(ev_str)
            #extract logical for all channels            
        
            #Do mapping if required
            if self.mapping_dict:
                evs.filter_by_mapping(self.mapping_dict)
            if len(evs)>0:
                #if nev==2: print 'nonISI', evs.get_tmadev()[0:5]
                evs.normalize_tm()
                #if nev==2: print 'nonISInorm', evs.get_tmadev()[0:5]
                evs.set_isi()
                if sys.byteorder == 'big':
                    stimByteStream = evs.get_tmadev().byteswap().tostring()
                else:
                    stimByteStream = evs.get_tmadev().tostring()
                #if nev==2: print 'ISI', evs.get_tmadev()[0:5]
                self.put_buffer(stimByteStream)

            return True  # queue is not empty
        else:
            return False  # queue is now empty

    def run(self):
        while self.finished.isSet() != True:
            t0 = time.time()

            self.recvlock.acquire()
            self.fetch_and_relay()
            self.recvlock.release()

            #Fastest frame rate should be fT. Wait in case we've been faster
            t_left = self.fT - (time.time() - t0)
            if t_left > 0:
                time.sleep(t_left)

class AEXRelay(threading.Thread):
    def __init__(self, buffer, host_stim='localhost', port_stim=50002, autostart=True):
        threading.Thread.__init__(self)
        self.stim_buffer = buffer
        self.daemon = True  # The entire Python program exits when no alive non-daemon threads are left
        print "Connecting to " + host_stim
        self.host_stim = host_stim

        self.stim_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.stim_sock.connect((self.host_stim, port_stim))

        #Build decoder functions
        if autostart:
            self.start()

    def relay(self):
        self.stim_sock.send(self.stim_buffer.get(block=True))

    def run(self):
        while True:
            self.relay()


