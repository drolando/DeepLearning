DeepLearning
============
Development of a deep convolutional neural network for image recognition using the sparse filtering technology.

The research paper published by Stanford University can be found at: [http://cs.stanford.edu/~jngiam](http://cs.stanford.edu/~jngiam/papers/NgiamKohChenBhaskarNg2011.pdf)

The source code will be published as soon as possible.

Firstly install mkl intel library!

### INSTALL
#### UBUNTU 14.04

```Shell
sudo apt-get install python-dev
sudo apt-get install libfreetype6-dev
sudo apt-get install python-scipy python-matplotlib ipython
sudo apt-get install libblas-dev
sudo easy_install networkx
sudo easy_install numexpr
sudo apt-get install python-skimage
sudo apt-get install python-pydot
sudo apt-get install python-mpi python-mpi4py
--->> sudo -E pip install --upgrade sklearn
```

#### MAC OS X 10.9

```Shell
export CFLAGS=-Qunused-arguments
export CPPFLAGS=-Qunused-arguments

sudo -E pip install --upgrade numpy
sudo -E pip install --upgrade scipy
sudo -E pip install --upgrade cython
sudo -E pip install --upgrade scikit
sudo -E pip install --upgrade matplotlib
sudo -E pip install --upgrade pydot
brew install graphviz
sudo -E pip install --upgrade pillow
sudo -E pip install --upgrade scikit-learn
```
