#! /suer/bin/env python
'''
@author: stober
'''

from distutils.core import setup

setup(name='svm',
      version='0.1',
      description='Support Vector Machines',
      author='Jeremy Stober',
      author_email='stober@gmail.com',
      package_dir={'svm':'src'},
      packages=['svm'],
      )
