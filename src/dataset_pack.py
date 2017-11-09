""""
   The aim of this program is to convert text data files containing
   NN inputs and outputs into binary data file. Inputs are "per frame"
   features extracted from audio signal. Outputs are vectors where each 
   element corresponds to a phoneme, that emitted a frame.
   The program loads the text file containing all training, development
   and testing data segmented by **** tags into isolated utterances 
   and stores it as binary numpy arrays. 
   
   Two arrays are stored: features array and classes array. 
   3-dimensional feature array is indexed in the following way:
      features[ utterance_index, frame_index, scalar_feature_index ]    
   3-dimensional NN output array is indexed in the following way:
      outputs[ utterance_index, frame_index, phoneme_index ]       
"""
   
import numpy as np
import sys
import re
import asr

# Constants definition
EOU_TAG = "****";

# ===========================================================================================

def get_utter( f_in, utters ):
    """
       It creates python  sequence of pairs where each pair represents single utterance
       The pair consists of two numpy 2D arrays where rows correspond to frames and columns 
       correspond to features/classes
    """
    
    # initialize utter block - sequences of frames for single utterance
    f_block_tmp = [];
    c_block_tmp = [];
    
    # create numpy arrays for single input/feature and output/classes vectors
    frm_f = np.zeros( asr.FEATURE_COUNT );
    frm_c = np.zeros( asr.CLASS_COUNT, dtype=np.int16 );
    
    while ( True ):
        line = f_in.readline();
        
        # remove leading/trailing whitecahrs
        line = line.rstrip();
        line = line.lstrip();        
        
        if ( len(line) == 0 ):
            # EOF was encountered - we can safely exit because EOL is always preceeded by ****
            return False;
            
        if ( line[0:4] == EOU_TAG ):
            # End-of-utterance tag detected; prepare the pair representing an utterance and add it to collection
            f_block = np.array( f_block_tmp );
            c_block = np.array( c_block_tmp );
            utters.append( [f_block, c_block] );
            # print "Completed: " + line[4:];
            f_block_tmp = [];
            c_block_tmp = [];
            
            # single call prepares only single utterance - so now it returns
            return True;
            
        # this is the next data line
        elems = re.split( "\ +", line );
        for i in range( asr.FEATURE_COUNT ):
            frm_f[i] = float( elems[i] )
        for i in range( asr.CLASS_COUNT ):
            frm_c[i] =  int( elems[i + asr.FEATURE_COUNT] );
        # append in/out vectors to the block of current utterance    
        f_block_tmp.append( np.copy(frm_f) );
        c_block_tmp.append( np.copy(frm_c) );
     
    # Should never happen - just for the case
    return False;

# ===========================================================================================   
    
def save_utters( utters, out_file ):    
    """
       It saves features and classes vectors as two 3D arrays.
       Indices are { uterance_ind, frame_ind, feature/class_ind }
    """
    frm_cnt = 0;
    for i in range( len(utters ) ):
       frm_cnt = frm_cnt + len(utters[i][0]);
    print "Total frames: " + str(frm_cnt );
    features = [ utters[i][0] for i in range( len(utters) ) ];
    classes  = [ utters[i][1] for i in range( len(utters) ) ];
    np_features = np.array( features );
    np_classes  = np.array( classes );

    f = open( out_file, "wb" );
    np.savez_compressed( f, a=np_features, b=np_classes );
    f.close();

# =============================================================================      
   
def load_utters( in_file ):    
    data = np.load( in_file );
    return [ data['a'], data['b'] ];

# =============================================================================
    
def test( in_file, utters ):
   lf, lc = load_utters( out_file );

   print "Features restored";
   print lf;

   print "Classes restored";
   print lc;
   print utters;  
    
# =============================================================================
# main()
# =============================================================================   

def main( params ):

    in_file, out_file = params

    # Each uttter is a pair of ndarrays: containing 
    # features and classes for subsequent frames. Each
    # array has as many rows as the number of frames. 
    # Number of columns is equal to features and classes count
    utters = [];

    # ==============================================================
    # First line contain feature and class count (per frame)
    # ==============================================================
    line = f_in.readline();
    line  = asr.clipLine( line );
    counts = re.split( "\ +", line );
    asr.FEATURE_COUNT = int( counts[0] );
    asr.CLASS_COUNT = int( counts[1] );
    print "Feature count " + str( asr.FEATURE_COUNT );
    print "Class count " + str( asr.CLASS_COUNT );

    proc_ut_cnt = 0;
    while ( get_utter( f_in, utters ) ):
        proc_ut_cnt = proc_ut_cnt + 1;
        
    print "Count of utterences: " + str( len(utters) )
    f_in.close();

    save_utters( utters, out_file );
    # test( out_file, utters );


    if ( len( sys.argv ) < 3 ):
        print "Missing params";
        sys.exit( 1 );

    # Each uttter is a pair of ndarrays: containing 
    # features and classes for subsequent frames. Each
    # array has as many rows as the number of frames. 
    # Number of columns is equal to features and classes count
    utters = [];

# =============================================================================
# Program
# =============================================================================   
    
if __name__ == '__main__':
    if ( len( sys.argv ) < 3 ):
        print "Usage:"
        print "   dataset_pack <input_file> <output_file>"
        sys.exit( 1 )
    
    main( (sys.argv[1], sys.argv[2] )
