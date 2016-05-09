#!/usr/bin/python

import sys

if __name__=='__main__':

    if len(sys.argv)!=3:
        print("Usage is: 'python "+__file__+" hostname channel'")
        sys.exit(0)
    import pyNCS
    import pyNCS.AerViewer
    setup=pyNCS.Setup('/home/emre/ini/experiments/chip_experiments/common/setups/sac_setuptype.xml')
    setup.load('/home/emre/ini/experiments/chip_experiments/common/setups/sac_setup.xml', offline = True, prefix = "/home/emre/ini/experiments/chip_experiments/common/")
    setup.apply()
    print("Starting AerViewer on " + sys.argv[1])
    pyNCS.AerViewer.Aer2DViewer(int(sys.argv[2]),host=sys.argv[1], dims=(64,32)).show()


