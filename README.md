DeepLearning
============
[![Build Status](https://travis-ci.org/drolando/DeepLearning.svg?branch=master)](https://travis-ci.org/drolando/DeepLearning)
[![Coverage Status](https://coveralls.io/repos/drolando/DeepLearning/badge.png?branch=master)](https://coveralls.io/r/drolando/DeepLearning?branch=master)
Development of a deep convolutional neural network for image recognition using the sparse filtering algorithm.

The research paper published by Stanford University can be found at: [http://cs.stanford.edu/~jngiam](http://cs.stanford.edu/~jngiam/papers/NgiamKohChenBhaskarNg2011.pdf)

### HOW TO RUN THE EXAMPLE
```
export PYTHONPATH='path/to/the/directory/where/you/downloaded/the/code'
cd sparse/demos/sparse_filtering

# Training phase
python demo_sparse.py --train [--unsupervised | --supervised | --all] [--input "/path/to/train.txt"] [--model "model_name"]

# Execution phase
python demo_sparse.py [--input path/to/train.txt] [--model "model_name"]
python demo_sparse.py [--input path/to/val.txt] [--model "model_name"]
python ../../../util/tsne_python/tsne.py
cd sparse/util/plibsvm-3.18
./gen.sh [-o "output_folder"] [-n "network_folder"]
```

## INPUT DATA
The zip file with all the training data, validation data and annotations taken from the TRECVID dataset can be downloaded from [Amazon S3](https://s3.amazonaws.com/deep_learning/data.zip).
The zip file with the images of cats, birds, dogs and lamps taken from the ImageNet dataset can be downloaded from [Amazon S3]()

All the input files (train.txt and val.txt) must be written with the following format:

image_path        label  
dog/dog_001.png   0  
cat/cat_002.png   0  
bird/dird_001.png 0  

To generate these files for the ImageNet dataset you can use the 'gen_train.sh' and 'gen_val.sh' scripts available in the folder 'sparse/util/_data'.
To generate the input files for the trecvid dataset you can use the 'gen_data.py' script in the folder 'sparse/util/_data/trecvid'.
```
./gen_train.sh  num_of_desired_images
./gen_val  num_of_desired_images

python gen_data.py  num_of_train_images  num_of_test_images
```

### INSTALL
Firstly install mkl intel library!
(Building numpy, scipy and numexpr from source will give a speed increase with respect to the precompiled versions downloaded through apt-get.)

#### UBUNTU 14.04

```Shell
sudo apt-get install python-dev
sudo apt-get install libfreetype6-dev
sudo apt-get install python-numpy
sudo apt-get install python-scipy python-matplotlib ipython
sudo apt-get install python-numexpr
sudo apt-get install libblas-dev
sudo easy_install networkx
sudo apt-get install python-skimage
sudo apt-get install python-pydot
sudo apt-get install python-mpi python-mpi4py
sudo -E pip install --upgrade sklearn
```

#### MAC OS X 10.9

```Shell
export CFLAGS=-Qunused-arguments
export CPPFLAGS=-Qunused-arguments

sudo -E pip install --upgrade numpy
sudo -E pip install --upgrade gfortran
sudo -E pip install --upgrade scipy
sudo -E pip install --upgrade cython
sudo -E pip install --upgrade scikit
sudo -E pip install --upgrade matplotlib
sudo -E pip install --upgrade pydot
brew install graphviz
sudo -E pip install --upgrade pillow
sudo -E pip install --upgrade scikit-learn
```
