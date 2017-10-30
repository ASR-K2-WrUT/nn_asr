#!/usr/bin/env python

"""
This source file presents an example how to use the datasets prepared with TIMIT-like
acoustic corpora in training and testing neural networks aimed phoneme recognition or
state probability estimation in HMM-NN hybrid ASR recognition.

This demo is based on the code published as a part of Lasagne tutorial. As in the 
case of the primary code, it is mainly focused on loading ans usage of ASR datasets
instead of focusing on writing maximally modular and reusable code. 

Dataset building modules compatible with the dataset format used herein can be found 
at GitHub: https://github.com/ASR-K2-WrUT/nn_asr

"""

from __future__ import print_function

import sys
import os
import os.path as pth
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import asr
import matplotlib.pyplot as plt
from lasagne.nonlinearities import LeakyRectify

rows_cnt = -1;
channels_cnt = -1;
features_cnt = -1;
output_size = -1;
mean_horizon = 5;
phone_list_file = "all_phones_grp.txt"


# ====================================================================================
# ====================================================================================
# Functions that load or prepare data fro NN training/testing
# ====================================================================================
# ====================================================================================

def load_dataset_crafted( train_fn="", valid_fn="", test_fn="" ):
    """
       The function loads train/development/test datasets form files specified as parameters. 
       If a filename is an empty string then the corresponding dataset is not loaded       
       For each actually loaded dataset a triple is created: 
                         (input_array, output_array, frame_position_array )
       which is returned by the function.  
       
       Input_array is a 4D array containing features being used as inputs to a NN. The first 
       dimension of input array corresponds to dataset elements which we call "recognizable objects". 
       The recognizable object corresponds to a single frame obtained from a audio file containing 
       the recorded speech sample (utterance). Features of the recognizable object are features 
       extracted from a subsequence of frames surrounding the one represented by the object. Using 
       features not only from the single frame but also from its neighbours makes it possible
       to take context into account.
       
       A 3D array constituted by remaining dimensions contains features of a recognizable object.
       Output_array is an 1D array containing true phone indices of a recognizable object. 
    
       Frame_position_array contains indices of recognizable objects within the utterances, 
       they come from. This information is useful is boundaries of utterances need to be
       determined.
    """
    
    global rows_cnt;
    global features_cnt;
    global output_size;
    global channels_cnt;
    
    def reorder_outputs( y_data, lookup ):
        """
           This function changes order of NN outputs. The order output 
           may be important if the outputs of NN are then used as inputs 
           to next stage convolutional NN.
           lookup[i] is the position of i-th output in the new vector of outputs.
           The reordering should be carried out once at the begining of the training process
           because for the next stages the reordered data will be stored in desired order
           The operation od executed in-place
        """
        for i in xrange(len(y_data)):
            y_output[i] = lookup[ y_output[i] ]
        return y_data
            
    def load_clarin(filename):
        f = open( filename, "rb" );
        fpair = np.load( f );
        inputs = fpair["a"];
        outputs = fpair["b"];   
        if ( 'c' in fpair ):
            positions = fpair["c"];                     
            dataset = ( inputs, outputs, positions );      
        else:
            dataset = ( inputs, outputs, None );      
        f.close();
        return dataset   
   
    if ( valid_fn != "" ):
       print( "Loading validation set: " +  valid_fn );        
       X_val, y_val, pos_val = load_clarin( valid_fn );
       output_size = np.max( y_val ) + 1;          
       tr_size, channels_cnt, rows_cnt, features_cnt = X_val.shape;       
    else:
       X_val = None
       y_val = None
       pos_val = None
    
    if ( test_fn != "" ):
       print( "Loading test set: " +  test_fn );        
       X_test, y_test, pos_test   = load_clarin( test_fn );    
       tr_size, channels_cnt, rows_cnt, features_cnt = X_test.shape;              
    else:
       X_test = None
       y_test = None
       pos_test = None

    if ( train_fn != "" ):
       print( "Loading training set: " +  train_fn );    
       X_train, y_train, pos_train = load_clarin( train_fn );
       output_size = np.max( y_train ) + 1;          
       tr_size, channels_cnt, rows_cnt, features_cnt = X_train.shape;
       
       output_size = np.max( y_train ) + 1;   
   
       print( "Count of traininig samples  : {0:,d}".format(tr_size) );
       print( "Count of features per frame : %d x %d x %d " % ( channels_cnt, rows_cnt, features_cnt ) );
       print( "Count of NN outpts          : %d" % output_size );       
    else:
       X_train = None
       y_train = None
       pos_train = None         
    
    return X_train, y_train, pos_train, X_val, y_val, pos_val, X_test, y_test, pos_test
    
# ===========================================================================
   
def load_dataset_from_raw( train_fn, valid_fn, test_fn, ctx_width, fraction ):
    """
       This function loads train/development/test datasets from "raw" datafiles.
       Raw datafiles contain frame data obtained directly from feature extraction 
       procedures. The raw datafile contains data obtained from the whole available 
       corpus of recorded speech samples. The data are organized in two arrays: 
       input data array and output data array. Input data array is the (NxM) array 
       where rows correspond to frames in audio streams and columns correspond to 
       features extracted from a single frame.
       The function builds data that can be used as inputs/outputs of trained/tested 
       NN. The role of the procedure is to:
       - select the subset of data in a row file corresponding to a fraction specified by 
         "fraction" parameter;
       - extend feature vectors of recognizable objects by merging features from adjacent
         subsequent frames, where the number of concatenated feature vectors is determined
         by "ctx_width" parameter. This parameters defines "context radius", i.e. the number 
         of concatenated vectors is equal to 2*ctx_withd+1
    """
    global rows_cnt;
    global features_cnt;
    global output_size;    
   
    print( "Loading training set: " +  train_fn );    
    X_train, y_train = asr.loadRawData( train_fn, ctx_width, fraction )
    print( "Loading validation set: " +  valid_fn );    
    X_val,   y_val   = asr.loadRawData( valid_fn, ctx_width, 1.0 )
    print( "Loading test set: " + test_fn );    
    X_test,  y_test  = asr.loadRawData( test_fn, ctx_width, 1.0 )

    tr_size, layer_cnt, rows_cnt, features_cnt = X_train.shape;
    output_size = np.max( y_train ) + 1;
    
    print( "Count of training samples   : {0:,d}".format(tr_size) );
    print( "Count of features per frame : %d x %d x %d " % ( layer_cnt, rows_cnt, features_cnt ) );
    print( "Count of NN outputs         : %d" % output_size );
    
    return X_train, y_train, X_val, y_val, X_test, y_test
   
# ===========================================================================

def prepare_2nd_order_data( X_data, y_data, L, src_fname, nnout_fn, stride=1 ):
    """
       This is a function that prepares data set where trained NN outputs are gathered
       so that it can be used as the input to 2nd order NN. The data are prepared for 
       train,validation and test sets. The format is analogous as in the case of
       primary input data.
    """

    print ( "Building 2nd order features: {0}".format(src_fname) )
    print( "   Recognizing on training set" );
    
    output_size = np.max( y_data ) + 1;
    tr_size, chan_cnt, whole_ctx, ft_sublen = X_data.shape
    
    all_outputs = np.zeros( (tr_size, output_size ) );
    all_outputs_ext = np.zeros( ( tr_size, 1, whole_ctx, output_size ) );
    ctx_width = (whole_ctx - 1)/2
    
    batch_size = 500
    opos = 0;
    for batch in iterate_minibatches(X_data, y_data, batch_size, shuffle=False):
        inputs, targets = batch
        outputs = nnout_fn( inputs )
        all_outputs[opos:opos+batch_size] = outputs
        opos = opos + batch_size

    print( "   Context assembling" ); 
    outputs_avg = np.zeros( ( tr_size, output_size ) );	
    whole_ctx_rev = 1.0 / (2*ctx_width + 1)
    for pos in range( tr_size ):
        lin_pos = 0;
        for ctx in range( -ctx_width*stride, ctx_width*stride+1, stride ):
            src_pos = pos + ctx
            if ( pos + ctx < 0 ):
                src_pos = 0;
            else:
                if ( pos + ctx >= tr_size ):
                    src_pos = tr_size - 1
                else:
                    if ( ctx < 0 ):
                        if (L[pos] < -ctx ):
                            src_pos = pos - L[pos];                   
                    if ( ctx > 0 ):   
                        if ( L[pos + ctx] < L[pos] ):
                            src_pos = pos + ctx - L[pos + ctx] - 1;
            if (( ctx >= -2) and ( ctx <= 2)):	
                outputs_avg[pos] = outputs_avg[pos] + (whole_ctx_rev * all_outputs[src_pos])
            all_outputs_ext[pos][0][lin_pos] = all_outputs[src_pos]
            lin_pos = lin_pos + 1                                   
        
    print( "   Storing in a file" );            
    in_file = src_fname;
    t = pth.splitext( in_file );
    out_file = t[0] + ".2nd" + t[1]
    print( "   Storing in a file: " + out_file );  
    f_out = open( out_file, "wb" );        
    np.savez( f_out, a=all_outputs_ext, b=y_data, c=L );             
    
    del all_outputs
    del all_outputs_ext

    return outputs_avg
    
# ===========================================================================

def prepare_reco_pairs( X_data, y_data, nnout_fn, batch_size=500 ):
    """
       This function prepares pairs (y_recognized, y_expected) for 
       all testing objects
    """
    y_reco = np.zeros( len(y_data), dtype=np.int16);
    opos = 0;
    for batch in iterate_minibatches(X_data, y_data, batch_size, shuffle=False):
        inputs, targets = batch
        outputs = nnout_fn( inputs )
        y_reco[opos:opos+batch_size] = np.argmax(outputs, axis=1 );
        opos = opos + batch_size
    return zip( y_reco, y_data)
    
    
# ====================================================================================
# ====================================================================================
# Functions that build NN models. 
#    These scripts supports two types of models: simple MLP and NN with the single 
#    convolution layer. For each one, we define a function that takes a Theano 
#    variable representing the input and returns the output layer of a neural network 
#    model built in Lasagne.
# ====================================================================================
# ====================================================================================

def build_mlp(input_var=None):
    """
       This function is an adaptation of the code borrowed from Lasagne
       usage examples (https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py)
       This creates an MLP of two hidden layers, followed by a softmax output layer 
       with the number of outputs specified by "outpout_size" global variable.  
       It applies 20% dropout to the input data and 50% dropout to the hidden layers
       Number of channels and the dimensions of channel data are determined by global 
       variables: rows_cnt, features_cnt.
    """

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel) and linking it to the given Theano variable:
    # `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, rows_cnt, features_cnt),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=2500,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another hidden layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=2200,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)   
   
    # Finally, we'll add the fully-connected output layer of softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=output_size,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

# =================================================================

def build_cnn(input_var=None, num_hidden=3, num_units=2800):
    """
       Function that builds the model of the convolutional NN
    """

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, channels_cnt, rows_cnt, features_cnt),
                                        input_var=input_var)

    # Convolutional layer with 128 kernels of size 11,7. 
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(11, 7),
            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.05),
            W=lasagne.init.GlorotUniform())
    
    # A fully-connected layer with 50% dropout on its inputs:
    for i in range( num_hidden )
        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=num_units,
                nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01))
    
    # And, finally, the output layer with 20% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.2),
            num_units=output_size,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

# ====================================================================================
# ====================================================================================
# Batch iterator 
# ====================================================================================
# ====================================================================================

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
       This is just a simple helper function iterating over training data in
       mini-batches of a particular size, optionally in random order. It assumes
       data is available as numpy arrays. For big datasets, you could load numpy
       arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
       own custom data iteration function. For small datasets, you can also copy
       them to GPU at once for slightly improved performance. This would involve
       several changes in the main program, though, and is not demonstrated here.
       Notice that this function returns only mini-batches of size `batchsize`.
       If the size of the data is not a multiple of `batchsize`, it will not
       return the last (remaining) mini-batch.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

        
# ====================================================================================
# ====================================================================================
# main() function
#    Everything else will be handled in our main program now. We could pull out
#    more functions to better separate the code, but it wouldn't make it any
#    easier to read.
# ====================================================================================
# ====================================================================================

def main( argv ):
    """
       The whole program trains NN and tests its performance
    """
    
    model = argv[1]
    num_epochs = int(argv[2])
    train_fname = argv[3], 
    valid_fname = argv[4], 
    test_fname  = argv[5]    
    
    use_raw = False;
    fraction = 1.0;
    ctx_width = 5;
    if ( len( argv ) > 6 ):
       if ( argv[6][0] == "B" ):
            use_bulk = True;

        if ( len( argv ) > 8 ):
            ctx_width = int( argv[7] );
            fraction = float( argv[8] ) * 0.01;
            if ( fraction > 1.0 ):
                fraction = 1.0;    

    # Load the dataset                
    positional_data_ready = False;
    if ( use_raw ):
       print( "Loading from raw data: context %d   fraction %4.2f" % ( ctx_width, fraction ) );    
       X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_from_bulk( train_fname, valid_fname, 
                                                                                test_fname, ctx_width, fraction )
    else:
       print( "Loading from preselected sets" );    
       X_train, y_train, pos_train, X_val, y_val, pos_val, X_test, y_test, pos_test = load_dataset_crafted( 
                                                                                         train_fname, valid_fname, "" )
       positional_data_ready = True;
      
    acc_history = [];
    acc_history_avg = [];
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    updates = lasagne.updates.nesterov_momentum(
              loss, params, learning_rate=0.01, momentum=0.9)
    # updates = lasagne.updates.adam(loss, params)
    
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True )

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    nnout_fn = theano.function( [input_var], test_prediction, allow_input_downcast=True)
    
    # Finally, launch the training loop.
    print("Starting training...")
    max_acc = 0.0;
    last_impr = 0;
    acc_mean = np.zeros( mean_horizon );
    
    train_loss_history = [];
    test_loss_history = [];    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        val_result = val_acc / val_batches;
        acc_mean[ epoch % mean_horizon ] = val_result;
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_result * 100))
        print("  mean accuracy:      \t\t{:.2f} %".format(
            np.mean(acc_mean) * 100))

        if ( val_result > max_acc + 0.0001 ):
           print ( "  improvement by %5.2f" % ( (val_result - max_acc) *100,) );        
           max_acc = val_result;
           last_impr = epoch;
           if ( epoch > 0 ) or ( epoch == 0 ):
               np.savez('model.npz', *lasagne.layers.get_all_param_values(network));
               print( "  net saved" );
        if ( epoch - last_impr > 40 ):
           break;          
       
        if ( epoch >= mean_horizon ):
            acc_history.append( val_result * 100 );
            acc_history_avg.append( np.mean(acc_mean) * 100 );

            train_loss_history.append( train_err / train_batches );
            test_loss_history.append( val_err / val_batches );
            
            xpos = np.argmax( acc_history )             
            plt.figure(figsize=[6,6])
            plt.grid( True );
            plt.ylabel( "Accuracy" );
            plt.xlabel( "Epochs" );
            plt.plot( acc_history, color = 'r', label="Acc" );
            plt.plot( acc_history_avg, color = 'b', label= "Avg_acc" );
            plt.legend();
            plt.axvline(x=xpos, color="black", ls="dashed", ymax=acc_history[xpos] )           
            t = pth.splitext( sys.argv[3] );
            out_file = t[0] + ".acc.png"
            plt.savefig( out_file );
            plt.clf()
            plt.cla()
            plt.close()
            
            plt.figure(figsize=[6,6])
            plt.grid( True );
            plt.ylabel( "Loss" );
            plt.xlabel( "Epochs" );
            plt.plot( train_loss_history, color = 'r', label="Train loss" );
            plt.plot( test_loss_history, color = 'b', label= "Validation loss" );
            plt.legend();
            xpos = np.argmin( train_loss_history )
            plt.axvline(x=xpos, color = 'r', ls="dashed" )            
            xpos = np.argmin( test_loss_history )
            plt.axvline(x=xpos, color = 'b', ls="dashed" )  
            
            t = pth.splitext( sys.argv[3] );
            out_file = t[0] + ".loss.png"
            plt.savefig( out_file );
            plt.clf()
            plt.cla()
            plt.close()                        
                   
    # After training, we compute and print the test error:
    with np.load('model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)    

    del X_val
    del y_val
    del pos_val
    
    X_train_dummy, y_train_dummy, pos_train_dummy, \
        X_val_dummy, y_val_dummy, pos_val_dummy, \
        X_test, y_test, pos_test = load_dataset_crafted( "", "", sys.argv[5] )    
    
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    print("  best at iter:\t\t{}".format( last_impr ) );
       
    # Save accuracy statistics on test data   
    in_file = sys.argv[3];
    t = pth.splitext( in_file );
    out_file = t[0] + ".rep.txt"
    results = prepare_reco_pairs( X_test, y_test, nnout_fn, batch_size=500 );
    asr.save_stats( asr.conf_mtr( results ), out_file, phone_list_file );       
    
    out_file = "test_history.rep.txt"
    with open(out_file, "a") as file:
        file.write( "Experiment: {} {} {}\n".format( sys.argv[3], sys.argv[4], sys.argv[5] ) )
        file.write( "  test loss:\t\t\t{:.6f}\n".format(test_err / test_batches) )
        file.write( "  test accuracy:\t\t{:.2f} %\n".format(test_acc / test_batches * 100))    

    # If data determining utterance boundaries are available the prepare datafiles
    # stacked NNs training    
    if ( positional_data_ready ):  
        stride = 1 

        prepare_2nd_order_data( X_train, y_train, pos_train, sys.argv[3], nnout_fn );
        del X_train
        del y_train
        del pos_train

        X_train_dummy, y_train_dummy, pos_train_dummy, X_val, y_val, pos_val, X_test_dummy, \
            y_test_dummy, pos_test_dummy = load_dataset_crafted( "", sys.argv[4], "" )
        prepare_2nd_order_data( X_val, y_val, pos_val, sys.argv[4], nnout_fn, stride );
        del X_val
        del y_val
        del pos_val
        
        X_train_dummy, y_train_dummy, pos_train_dummy, X_val_dummy, y_val_dummy, pos_val_dummy, \
            X_test, y_test, pos_test = load_dataset_crafted( "", "", sys.argv[5] )
        outputs_avg = prepare_2nd_order_data( X_test,  y_test, pos_test, sys.argv[5], nnout_fn, stride );
        test_acc_avg = np.mean(np.equal(np.argmax(outputs_avg, axis=1), y_test) ) * 100.0
        with open(out_file, "a") as file:
           print ( "Accuracy by everage: %6.2f" % test_acc_avg )	
           file.write("  test averaged:\t\t{:.2f} %\n".format( test_acc_avg	) )

        del X_test
        del y_test
        del pos_test   
    
# ====================================================================================
# ====================================================================================

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or (len(sys.argv) < 6 ):
        print("Demo program showing how to use CLARIN_NN dataset for NN training using Lasagne.")
        print("Usage: %s <model> <epochs> <train_set_file> <devel_set_file> <test_set_file> [B <context> <fraction>]" % sys.argv[0])
        print()
        print("  <model>: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("           'cnn' for a simple Convolutional Neural Network (CNN).")
        print("  <epochs>: number of training epochs to perform (default: 500)")
        print("  <train_set_file> " )
        print("  <devel_set_file> " )
        print("  <test_set_file>: dataset files prepared with nn_reshape.py" )
        print("  <context>:       frame context radius, used only if import from raw dataset ")
        print("  <fraction>:      percentage of raw train dataset to be used for training ")
    else:
        main( sys.argv )
