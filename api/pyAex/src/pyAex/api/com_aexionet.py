from pyNCS.api.ComAPI import *
import pyAex.aexio as aexio
from pyAex.aexutils import *
import getpass

XIOOUT_FN = '/tmp/{0}_xio_out'.format(getpass.getuser())
XIOIN_FN = '/tmp/{0}_xio_in'.format(getpass.getuser())
LOCALHOST = 'localhost'

#Has a single client.
#Instanciated object can be reused


class Communicator(BatchCommunicatorBase):
    '''
    This class implements the BatchCommunicator API defined by NCS.
    It uses the *xio* executable on the target host, and transfer input and output event streams through secure shell (scp).

    Inputs:
    *host:* the hostname of the computer where the AEX board is attached (default: localhost). If localhost is chosen, then ssh commands are *not* used.
    *devnum:* id number of the AEX device file, appended to '/dev/aerfx2_'. (default: 0)
    *user:* of hostname is other than localhost, *user* is used to connect via secure shell (ssh) to the host computer

    Usage:
    >>> c = Communicator(host = 'localhost', devnum = 0)
    >>> c.run()

    '''
    def __init__(self, host=LOCALHOST, devnum=0, user=None, *args, **kwargs):
        self.host = host
        self.args = args
        self.kwargs = kwargs
        self.devname = '/dev/aerfx2_' + str(devnum)
        if user != None:
            self.user = user
        else:
            self.user = getpass.getuser()
        BatchCommunicatorBase.__init__(self)

    def _run_cmd(self, duration):
        duration_us = duration * 1000
        return 'xio MONSEQ {3} {0} <{1} >{2}'.format(self.devname, XIOIN_FN, XIOOUT_FN, duration_us)

    @doc_inherit
    def run(self, stimulus=None, duration=None, context_manager=None, **kwargs):
        self.stim(stimulus, duration, context_manager, **kwargs)
        return self._evs_postprocess()

    @doc_inherit
    def stim(self, stimulus=None, duration=None, context_manager=None, **stim_kwargs):
        self._evs_preprocess(stimulus)

        if context_manager == None:
            context_manager = empty_context

        if duration == None:
            duration = 0

        run_cmd = self._run_cmd(duration)
        if self.host != LOCALHOST:
            self.__ssh_send()
            self.__ssh_xio(run_cmd, context_manager)
            self.__ssh_get()
        else:
            self.__xio(run_cmd, context_manager)

    def _evs_preprocess(self, stimulus):
        evs_in = events(stimulus, 'p')
        return evs_in.get_tmadev().tofile(XIOIN_FN)

    def _evs_postprocess(self):
        try:
            data = np.loadtxt(XIOOUT_FN, dtype='float',
                 converters={0: float, 1: float.fromhex})
        except IOError, e:
            # The error is not explicit: it could be anything, so double check
            # that the file exists but that there are no events
            if is_file_empty(XIOOUT_FN):
                data = np.zeros([0, 2], dtype='float')
            else:
                raise

        evs_out = events(np.fliplr(
            data), 'p').get_adtmev()  # Slight overhead but safely sets the type
        return evs_out

    def __ssh_send(self):
        #Correct to xio input
        system('scp {2} {0}@{1}:{2}'.format(self.user, self.host, XIOIN_FN))

    def __ssh_xio(self, run_cmd, context_manager):
        with context_manager():
            system(
                'ssh -C {0}@{1} "{2}"'.format(self.user, self.host, run_cmd))

    def __xio(self, run_cmd, context_manager):
        with context_manager():
            system('{2}'.format(self.user, self.host, run_cmd))

    def __ssh_get(self):
        system('scp {0}@{1}:{2} {2}'.format(self.user, self.host, XIOOUT_FN))
