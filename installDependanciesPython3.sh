#!/bin/bash
# do not run this with sudo 
# Maybe just run the first one with sudo

apt-get -y install python3-dev python3-pip
# sudo apt-get -y install python3-tk

pip3 install --user numpy
pip3 install --user Theano==0.8.2
# pip install matplotlib
pip3 install --user matplotlib
pip3 install --user Lasagne==0.1
pip3 install --user dill
pip3 install --user pathos
pip3 install --user pyOpenGL
pip3 install --user keras
## Does not work in Python3
# pip3 install pyODE
pip3 install --user h5py
pip3 install --user python3-tk

