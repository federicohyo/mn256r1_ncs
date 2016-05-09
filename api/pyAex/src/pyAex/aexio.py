from aexutils import *
import pyNCS.pyST as pyST
import getpass

### Proper AEX functions


def AEXIO(
        stim=None,
        tDuration=1000,
        devnum="0",
        MonChannelAddress=None,
        SeqChannelAddress=None,
        context=None):
    '''
    AEXIO - Stimulate AER chip using AEX board and monitor its
    response using Daniel Fasnacht's C code.

    This script requires the C executable command "xio" to be in the system path

    see pyST.STChannelAddressing.exportAER to see how 'stim' can be specified

    *stim*  Stimulation of the chips. This can take several different forms: if SpikeList object given, channel 0 is stimulated. If a list is given, its length must be equal to the number of channels (may contain None). If dict is given, the key should be an integer defining the channel
    *tDuration*  Monitor time in milliseconds, works only if stim=None
    *devnum*  An int defining the device number of the Aex board
    *MonChannelAddress*  STChannelAddressing object that will be used for reading the AER output. If None, determined with pyST.getDefaultMonChannelAddress
    *SeqChannelAddress*  STChannelAddressing object that will be used for generating the AER input. If None, then determined with pyST.getDefaultSeqChannelAddress
    *context* A context manager wrapping the stimulation/monitoring.
    '''

    if MonChannelAddress == None:
        MonChannelAddress = pyST.getDefaultMonChannelAddress()
    if SeqChannelAddress == None:
        SeqChannelAddress = pyST.getDefaultSeqChannelAddress()

    devname = '/dev/aerfx2_' + str(devnum)
    tdir = '/tmp/'
    stimfile = tdir + 'aexstim' + str(
        devnum) + '_' + getpass.getuser() + '.txt'
    monfile = tdir + 'aexmon' + str(devnum) + '_' + getpass.getuser() + '.txt'

    if context == None:
        context = _WITHOUT_CONTEXT

    if stim != None:
        #Note: exportAER will check stim
        SeqChannelAddress.exportAER(
            stim, filename=stimfile, sep=" ", format='t')
    else:
    #Ugly workaround for monitor only
        stim = pyST.SpikeList([(0, 0), (0, int(tDuration))], [0])
        MonChannelAddress.exportAER(
            {0: stim}, filename=stimfile, sep=" ", format='t')

    with context():
        system('aexio ' + devname + ' >' + monfile + ' <' + stimfile)

    evs = np.loadtxt(input=monfile, sep=" ", isi=False, format='t')

    return evs
