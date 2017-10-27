from __future__ import print_function

import sys
import os
import os.path as pth
import time

import numpy as np



import asr


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
        y_data[i] = lookup[ y_data[i] ]
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
 
def main( argv ):
    all_outputs_ext, y_data, pos_data = load_clarin( argv[1] ); 
    f = open( asr.phone_lookup_fn, "rb" );
    fpair = np.load( f );
    lookup = fpair["a"];
    f.close()    
    
    print( "Lookup_size %d" % len(lookup) );
    for i in range( len(lookup) - 1 ):
        print ("%d    %d" % ( i, lookup[i] ));
        
    reorder_outputs( y_data, lookup );
    
    out_file = argv[1]
    print( "Storing in a file: " + out_file );  
    f_out = open( out_file, "wb" );        
    np.savez( f_out, a=all_outputs_ext, b=y_data, c=pos_data );        
    f_out.close()
    return None;
    
if __name__ == '__main__':
    main( sys.argv )    