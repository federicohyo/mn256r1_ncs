# -*- coding: utf-8 -*-
try:
    from setuptools import setup, find_packages
except:
    print ''' 
setuptools not found.

On linux, the package is often called python-setuptools'''
    from sys import exit
    exit(1)


setup(name='pyMonster',
    version='0.99',
    description='',
    author='Fabio Stefanini, Federico Corradi',
    author_email='',
    url='',
    packages = find_packages(),
    #scripts = ['run_server_monster.py', 'run_client_monster.py'],
     )   

