#!/usr/bin/python

import sys

if __name__=='__main__':

    if len(sys.argv)!=3:
        print("Usage is: 'python "+__file__+" hostname channel'")
        sys.exit(0)

    import pyNCS
    import pyNCS.AerViewer
    from pyNCS.neurosetup import NeuroSetup

    #setup settings for zenzero
    nsetup = NeuroSetup(\
        setuptype='setups/fes_mapper_setuptype.xml',
        setupfile='setups/ifmem_fes_setup.xml',
        offline=False,
        prefix='./') # load zenzero setuptype

    print("Starting AerViewer on " + sys.argv[1])
    pyNCS.AerViewer.Aer1DViewer(int(sys.argv[2]),host=sys.argv[1]).show()




