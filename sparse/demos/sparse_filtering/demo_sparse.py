# coding=utf-8
# This 2 lines allow the network to be executed on a remote server without graphic interface.
import matplotlib
matplotlib.use('Agg')

from sparse import base
from sparse.util import smalldata, visualize
from sparse.layers import core_layers, convolution, fillers, regularization, sparsenet, sparsefiltering
from sparse.opt import core_solvers
from sparse.layers.data.dataset import ImageDataLayer
import numpy as np
from matplotlib import pyplot
import sys, time, datetime
from sparse.util.tsne_python import tsne

NET_MODEL = 'unsupervised.sparsenet'

# NETWORK DEFINITION
# 
# Here all the layers are created and added to the network. Each layer must have a univoque
# name which is used to identify it.
# 
# To specify the input blobs of a layer you can use the parameter needs: it receives the names
# of the required blobs.
# Similarly the output blobs are set with the provides parameter; also in this case you should
# put the names of the new blobs.
#
# Each layer must have the group parameter: it is user for the layer wise training. All the 
# layers between a ConvolutionalLayer and the following one must be in the same group.
#
# All the layers must be added before calling net.finish().

net = sparsenet.SparseNet()
'''
    ImageDataLayer -- reads the input file, convert the images and then emits them as Blobs.
    train - path to the default location of train.txt
'''
net.add_layer(
    ImageDataLayer(
        name='input-layer',
        train='../../util/_data/train.txt'
        #train='../../util/_data/val.txt'
    ),
    group=0,
    provides=['data', 'labels']
)
'''
    forward - (10, 57, 57, 96)
'''
net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv1',
        num_kernels=96,
        ksize=11,
        stride=4,
        mode='same',
    ),
    group=1,
    needs='data',
    provides='conv1_cudanet_out'
)
'''
    applies ReLU activation function
    all _data < 0 become zero
    forward - (10, 57, 57, 96)
'''
net.add_layer(
    core_layers.ReLULayer(
        name='conv1_neuron',
        freeze=False
    ),
    group=1,
    needs='conv1_cudanet_out',
    provides='conv1_neuron_cudanet_out'
)
'''
    forward - (10, 28, 28, 96)
'''
net.add_layer(
    core_layers.PoolingLayer(
        name='pool1',
        mode='max',
        psize=3,
        stride=2
    ),
    group=1,
    needs='conv1_neuron_cudanet_out',
    provides='pool1_cudanet_out'
)
'''
    forward - (10, 28, 28, 96)
'''
net.add_layer(
    core_layers.LocalResponseNormalizeLayer(
        name='rnorm1',
        alpha=0.0001,
        beta=0.75,
        size=5,
        k=1 # in imageNet k=2
    ),
    group=1,
    needs='pool1_cudanet_out',
    provides='rnorm1_cudanet_out'
)

'''
    2 groups by 128 kernels --> 256
    forward - (10, 28, 28, 256)
'''

net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv2',
        num_kernels=256,
        ksize=5,
        stride=1,
        mode='same',
    ),
    group=2,
    needs='rnorm1_cudanet_out',
    provides='conv2_cudanet_out'
)
'''
    forward - (10, 28, 28, 256)
'''
net.add_layer(
    core_layers.ReLULayer(
        name='conv2_neuron',
        freeze=False
    ),
    group=2,
    needs='conv2_cudanet_out',
    provides='conv2_neuron_cudanet_out'
)
'''
    forward - (10, 14, 14, 256)
'''
net.add_layer(
    core_layers.PoolingLayer(
        name='pool2',
        mode='max',
        psize=3,
        stride=2
    ),
    group=2,
    needs='conv2_neuron_cudanet_out',
    provides='pool2_cudanet_out'
)
'''
    forward - (10, 14, 14, 256)
'''
net.add_layer(
    core_layers.LocalResponseNormalizeLayer(
        name='rnorm2',
        alpha=0.0001,
        beta=0.75,
        size=5,
        k=1
    ),
    group=2,
    needs='pool2_cudanet_out',
    provides='rnorm2_cudanet_out'
)
'''
    forward - (10, 14, 14, 384)
'''
net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv3',
        num_kernels=384,
        ksize=3,
        stride=1,
        mode='same',
    ),
    group=3,
    needs='rnorm2_cudanet_out',
    provides='conv3_cudanet_out'
)
'''
    forward - (10, 14, 14, 384)
'''
net.add_layer(
    core_layers.ReLULayer(
        name='conv3_neuron',
        freeze=False
    ),
    group=3,
    needs='conv3_cudanet_out',
    provides='conv3_neuron_cudanet_out'
)
'''
    2 groups by 192 kernels --> 384
    forward - (10, 14, 14, 384)
'''


net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv4',
        num_kernels=384,
        ksize=3,
        stride=1,
        mode='same',
    ),
    group=4,
    needs='conv3_neuron_cudanet_out',
    provides='conv4_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='conv4_neuron',
        freeze=False
    ),
    group=4,
    needs='conv4_cudanet_out',
    provides='conv4_neuron_cudanet_out'
)

net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv5',
        num_kernels=256,
        ksize=3,
        stride=1,
        mode='same',
    ),
    group=5,
    needs='conv4_neuron_cudanet_out',
    provides='conv5_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='conv5_neuron',
        freeze=False
    ),
    group=5,
    needs='conv5_cudanet_out',
    provides='conv5_neuron_cudanet_out'
)
net.add_layer(
    core_layers.PoolingLayer(
        name='pool5',
        mode='max',
        psize=3,
        stride=2
    ),
    group=5,
    needs='conv5_neuron_cudanet_out',
    provides='pool5_cudanet_out'
)
net.add_layer(
    core_layers.FlattenLayer(
        name='fc6_flatten',
    ),
    group=5,
    needs='pool5_cudanet_out',
    provides='_sparse_fc6_flatten_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc6',
        num_output=4096,
        has_bias=True,
    ),
    group=6,
    needs='_sparse_fc6_flatten_out',
    provides='fc6_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='fc6_neuron',
        freeze=False
    ),
    group=6,
    needs='fc6_cudanet_out',
    provides='fc6_neuron_cudanet_out'
)
net.add_layer(
    core_layers.DropoutLayer(
        name='fc6dropout',
        ratio=0.500000,
    ),
    group=6,
    needs='fc6_neuron_cudanet_out',
    provides='fc6dropout_cudanet_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc7',
        num_output=4096,
        has_bias=True,
    ),
    group=7,
    needs='fc6dropout_cudanet_out',
    provides='fc7_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='fc7_neuron',
        freeze=False
    ),
    group=7,
    needs='fc7_cudanet_out',
    provides='fc7_neuron_cudanet_out'
)
net.add_layer(
    core_layers.DropoutLayer(
        name='fc7dropout',
        ratio=0.500000,
    ),
    group=7,
    needs='fc7_neuron_cudanet_out',
    provides='fc7dropout_cudanet_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc8',
        num_output=4,
        has_bias=True,
    ),
    group=8,
    needs='fc7dropout_cudanet_out',
    provides='fc8_cudanet_out'
)
net.add_layer(
    core_layers.SoftmaxLayer(
        name='probs',
    ),
    group=8,
    needs='fc8_cudanet_out',
    provides='probs_cudanet_out'
)
net.add_layer(
    core_layers.KLDivergenceLossLayer(
        name='loss',
    ),
    group=8,
    needs=['probs_cudanet_out', 'labels']
)

net.finish()
visualize.draw_net_to_file(net, 'network.png')

# MAIN FUNCTION
#
# PARAMETERS
# --input: path to the train.txt file (default: ../../util/_data/train.txt)
# --model: path to the model file (default: unsupervised.sparsenet)
# --train --unsupervised: starts the unsupervised training phase. The trained network will be saved
#                         in the model file.
# --train --supervised: starts the supervised training phase. <<< NOT WORKING >>>
# --train --all: executes both unsupervised and supervised trainings.
#
# OUTPUT
# The unsupervised learning generates the network model and saves it in the NET_MODEL file. It creates
# also a file 'net.params' which contains the basic informations about the training process.
# The execution phase generates 12 files called plibsvm_$i.[train|val] which are the output of the 
# corresponding layers. These files are used to train and test the libsvm library.
# 

if '--input' in sys.argv[1:]:
    net.layers['input-layer'].input_file = sys.argv[sys.argv.index('--input') + 1]

if '--model' in sys.argv[1:]:
    NET_MODEL = sys.argv[sys.argv.index('--model') + 1]

if '--train' in sys.argv[1:]:
    # TRAINING
    if '--unsupervised' in sys.argv[1:] or '--all' in sys.argv[1:]:
        if '--continue' in sys.argv[1:]:
            net.load_from(NET_MODEL)

        print "num_images: ", net.layers['input-layer'].get_num_images()
        start = time.time()
        net.sparse_filtering()
        net.save(NET_MODEL, store_full=False)
        filters = net.layers['conv1'].param()[0].data()
        _ = visualize.show_multiple(filters.T)
        pyplot.savefig('features.png')
        end = time.time()

        fp = open('net.params', 'a')
        fp.write('SPARSE NET PARAMETERS -- %s\n'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        fp.write('Processed layers: %s\n'%(str(sparsenet.PROCESSED_LAYERS)))
        fp.write('Num training images: %d\n'%(net.layers['input-layer'].get_num_images()))
        fp.write('Patches per image: %d\n'%(sparsenet.PATCHES_PER_IMAGE))
        fp.write('Max iterations: %d\n'%(sparsenet.MAX_ITER))
        fp.write('Tolerance: %s\n'%(str(sparsefiltering.TOLERANCE)))
        fp.write('Num labels: %d\n'%(net.layers['input-layer']._num_labels))
        fp.write('Input file: %s\n'%(net.layers['input-layer'].input_file))
        fp.write('Net model: %s\n'%(NET_MODEL))
        fp.write('Time elapsed: %d:%d\n'%((end-start)/60, (end-start)%60))
        fp.write('\n\n')
        fp.close()



    if '--supervised' in sys.argv[1:] or '--all' in sys.argv[1:]: 
        net.load_from(NET_MODEL)

        print "###################################################################"
        print "  Calling solver"
        print "###################################################################"

        solver = core_solvers.SGDSolver(
            lr_policy='exp',
            base_lr=1,
            gamma=0.5,
            max_iter=100,
            disp=10)

        solver.solve(net)

else:
    # EXECUTION
    file_name = net.layers['input-layer'].input_file.split('/')[-1].split('.')[0]
    net.load_from(NET_MODEL)
    net.layers['fc8']._num_output = 2
    fp = open('feat.%s'%(file_name), 'w')
    fp_rnorm1 = open('plibsvm_1.%s'%(file_name), 'w')
    fp_rnorm2 = open('plibsvm_2.%s'%(file_name), 'w')
    fp_conv3 = open('plibsvm_3.%s'%(file_name), 'w')
    fp_conv4 = open('plibsvm_4.%s'%(file_name), 'w')
    fp_conv5 = open('plibsvm_5.%s'%(file_name), 'w')
    fp_flatten = open('plibsvm_6.%s'%(file_name), 'w')

    for i in range(net.layers['input-layer'].get_num_images()):
        print "\nprocessing image %d"%(i)
        top = [base.Blob(), base.Blob()]
        
        feat = net.predict(output_blobs=['rnorm1_cudanet_out', 'rnorm2_cudanet_out', 'conv3_neuron_cudanet_out', 
                    'conv4_neuron_cudanet_out', 'conv5_neuron_cudanet_out', '_sparse_fc6_flatten_out', 'labels',])
        
        label = np.where(feat['labels'][0] == 1.)[0][0]
        fp_rnorm1.write('%d '%(label))
        fp_rnorm2.write('%d '%(label))
        fp_conv3.write('%d '%(label))
        fp_conv4.write('%d '%(label))
        fp_conv5.write('%d '%(label))
        fp_flatten.write('%d '%(label))
        
        #first layer
        cnt = 1
        for val in feat['rnorm1_cudanet_out'].reshape((10, np.prod(feat['rnorm1_cudanet_out'].shape[1:])))[4]:
            fp_rnorm1.write("%d:%f "%(cnt, val))
            cnt += 1
        fp_rnorm1.write('\n')

        #second layer
        cnt = 1
        for val in feat['rnorm2_cudanet_out'].reshape((10, np.prod(feat['rnorm2_cudanet_out'].shape[1:])))[4]:
            fp_rnorm2.write("%d:%f "%(cnt, val))
            cnt += 1
        fp_rnorm2.write('\n')
    
    
        #third layer
        cnt = 1
        for val in feat['conv3_neuron_cudanet_out'].reshape((10, np.prod(feat['conv3_neuron_cudanet_out'].shape[1:])))[4]:
            fp_conv3.write("%d:%f "%(cnt, val))
            cnt += 1
        fp_conv3.write('\n')

        #forth layer
        cnt = 1
        for val in feat['conv4_neuron_cudanet_out'].reshape((10, np.prod(feat['conv4_neuron_cudanet_out'].shape[1:])))[4]:
            fp_conv4.write("%d:%f "%(cnt, val))
            cnt += 1
        fp_conv4.write('\n')


        #fifth layer
        cnt = 1
        for val in feat['conv5_neuron_cudanet_out'].reshape((10, np.prod(feat['conv5_neuron_cudanet_out'].shape[1:])))[4]:
            fp_conv5.write("%d:%f "%(cnt, val))
            cnt += 1
        fp_conv5.write('\n')

        #sixth layer
        cnt = 1
        for val in feat['_sparse_fc6_flatten_out'][4]:
            fp_flatten.write("%d:%f "%(cnt, val))
            fp.write("%f "%(val))
            cnt += 1
        fp_flatten.write('\n')
        fp.write('\n')
    fp.close()
    fp_rnorm1.close()
    fp_rnorm2.close()
    fp_conv3.close()
    fp_conv4.close()
    fp_conv5.close()
    fp_flatten.close()
    
