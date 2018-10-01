# Learn

This package contains all of the python code used for learning. The code is based
on Lasagne which is based on Theano.

## Dependancies

 1. sudo apt-get -y install liblapack3 liblapack-dev libblas3 libblas-dev gfortran libspqr1.3.1 libcholmod2.1.2 libmetis5 libmetis-dev libcolamd2.8.0 libccolamd2.8.0 libcamd2.3.1 libamd2.3.1 libx11-dev python-dev  
     A bunch of math libraries that will be needed
 2. pip install Theano  
     This is the backbone of the learning framework.
 3. pip install matplotlib  
       Plotting things is great, especially while things are running
 4. pip install Lasagne==0.1  
      Makes creating neural networks easier
 5. sudo pip install dill  
     This will allow for some cool multi-processing later
 6. sudo pip install pathos
	 This library is nice update to the python multiprocessing library allowing
for the pickeling of objects as well as other things.
 7. apt-get -y install  swig3.0  
	Swig is used to generate Python code that wraps the C++ code.
 8. sudo pip install h5py  
	This library is used to save and load data from files

## Install On Windows

  1. Install [Anaconda](https://www.continuum.io/downloads)
  1. Follow the [setup instruction for Theano](http://deeplearning.net/software/theano/install_windows.html#install-requirements-and-optional-packages)
   which are 
   ```
   	conda install numpy scipy mkl-service libpython m2w64-toolchain <nose> <nose-parameterized> <sphinx> <pydot-ng>
   ```



### For GPU training

 1. sudo apt-get install nvidia-cuda-toolkit nvidia-cuda-dev nvidia-modprobe  
	These libraries are needed to compile code for the GPU as well as to check what GPU devices are available

NOTE: Ran into this issue on Ubuntu 16.04 (https://github.com/Theano/Theano/issues/4425)
As a temporary workaround, I use the following hack:

    Add cmd.append('-D_FORCE_INLINES') just before p = subprocess.Popen( in the file nvcc_compiler.py


## Using The system

```
python3 trainModel.py --config settings/particleSim/PPO/PPO.json
```
	

## References

 1. https://github.com/Newmu/Theano-Tutorials
 2. https://github.com/spragunr/deep_q_rl
