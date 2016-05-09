# -*- coding: utf-8 -*-
from distutils.core import setup

setup(name='pyAex',
	version='0.1a',
	description='python AEX Stimulation Monitoring tool',
	author='Emre Neftci',
	author_email='emre@ini.phys.ethz.ch',
	url='',
	packages=['pyAex','pyAexServer','pyAex.stimulators','pyAex.api'],
    package_dir={'pyAex' : 'src/pyAex','pyAexServer' : 'src/pyAexServer'},
     )



