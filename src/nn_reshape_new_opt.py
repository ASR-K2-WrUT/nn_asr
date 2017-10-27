"""
   This program reads the array of individual utterances.
   data are stored in two arrays: containing features and vectorized classes
   the class vector contaiing UNK phoneme representing all pronemes
   that were removed from recognition in Mohamad paper (39 true phoneme groups plus UNK)
  """
import sys
import numpy as np
import asr
import random
import pickle
import os

# ========================================================================================

def prepare_for_nn( all_features, all_classes ):
    """" 
       Prepares data to be read by Nielsen NNs.
       It takes feature array containinig extended features (including context)
       and vectorized NN output and reshapes it according to format expected by NN.
       It converts rows being simple 1D arrays into 2D arrays [N,1]
       These two arrays are elements of tuple/pair passed via argument
    """
    # Modify classes for exclusion
    trainset_size, out_veclen = all_classes.shape
    trainset_size, feature_veclen = all_features.shape;    
    inputs  = [ np.reshape(all_features[x], (feature_veclen, 1)) for x in range(trainset_size) ]
    results = [ np.reshape(all_classes[y],  (out_veclen,     1)) for y in range(trainset_size) ]
    data = zip( inputs, results)
    return ( data )      

# ========================================================================================

def prepare_for_theano( all_features, all_classes, fraction ):
    """" 
       Prepares data to be read by Nielsen NNs.
    """
    # Modify classes for exclusion

    trainset_size, out_veclen = all_classes.shape
    trainset_size, feature_veclen = all_features.shape; 
    inputs  = [ all_features[x]            for x in range(trainset_size) ]
    outputs = [ np.argmax(all_classes[y])  for y in range(trainset_size) ]

    np_inputs = np.array( inputs, dtype=np.float );    
    np_outputs = np.array( outputs, dtype=np.int32 );
    print "Theano shape inputs/outputs: ";
    print np_inputs.shape;
    print np_outputs.shape;
    
    data = ( np_inputs, np_outputs )
    return ( data );     
      
    
# ========================================================================================     
           
def load_utters( in_file ):   
    """ Reads two numpy arrays containinig features and vectorized classses per frame
        Features are aggregated by utters, so both arrays are:
        features[ utters_cnt][ utter_len_in_frames ][ feature_cnt_per_frame ]
        classes array is organized in the same way
    """    
    data = np.load( in_file );
    return [ data['a'], data['b'] ];

# ========================================================================================  
    
def process_utter( features, classes, ctx, fraction, max_datasize  ):
   global current_af_pos;
   global all_features;
   # print features.shape;
   frm_cnt = len(features )
   dst_f_len = asr.FEATURE_COUNT * ( 2*ctx + 1 );
   feature_vec = np.empty( (2*ctx + 1 , asr.FEATURE_COUNT ) );
   # print all_features.shape;
   for i in range( frm_cnt ):
      if ( np.random.uniform() > fraction ):
         continue;
      dst_pos = 0;
      lin_pos = i - ctx;      
      for dst_pos in range( 2*ctx + 1 ):
         pos = max( lin_pos, 0 );
         pos = min( pos, frm_cnt - 1 );
         feature_vec[dst_pos] = features[pos];      
         dst_pos = dst_pos + 1;
         lin_pos = lin_pos + 1;
         
      lin_features = feature_vec.reshape ( (dst_f_len) );   

      all_features[ current_af_pos ] = lin_features;
      all_classes[ current_af_pos ] = classes[i];
      current_af_pos = current_af_pos + 1;
      if ( current_af_pos >= max_datasize ):
         break;
      
   return current_af_pos;
      
# ===================================================================================
# ===================================================================================
# main (argv )
# ===================================================================================      
# ===================================================================================
if ( len( sys.argv ) < 4 ):
    print "Usage:";
    print "    python nn_reshape <in_file> <out_file> <context_width> [<fraction>]\n"
    print "Context width - single side context - number of adjacent frames"
    print "Fraction - [0.0-1.0] determines fraction of data to be used"
    sys.exit( 1 );

    
# ===================================================================================
# Get and process parameters
# ===================================================================================
in_file = sys.argv[1];
out_file = sys.argv[2];
ctx_width = int( sys.argv[3] );

fraction = 1.0;
if ( len( sys.argv ) > 4 ):
   fraction = float( sys.argv[4] );
if ( fraction > 1.0 ):
   fraction = 1.0;
   
use_theano = False;
if ( len( sys.argv ) > 5 ):
    if( sys.argv[5] == "T" ):
        use_theano = True;
   
print "Fraction of " + str( fraction ) + " will be used"
print "Whole ctx width: " + str( 2*ctx_width+1);


# ===================================================================================
# Load data - a sequence of utterance data. Separation is necessary in order to avoid
# crossing utterance boundaries when building ftame context 
# ===================================================================================  
features, classes = load_utters( in_file );

# Compute total number of frames in all utters
frames_cnt = 0;
for i in range( len(features) ): 
   frames_cnt = frames_cnt + len (features[i] )
print "Frames count: ", str( frames_cnt );
print "Utters count: ", str( len(features) );

hrs, mins, sec, dur = asr.frms2hrs( frames_cnt )
max_datasize = min( int( frames_cnt * fraction ), frames_cnt );
print "Expected recognizables count: " + str(max_datasize);
print "Total duration: ", str( frames_cnt * 0.01 ), " sec.  (" + dur + ")";

# Get actual features and frames count
asr.FEATURE_COUNT = len( features[0][0] );
asr.CLASS_COUNT = len( classes[0][0] );
print "Features count: %d" % (asr.FEATURE_COUNT);
print "Class count: %d" % (asr.CLASS_COUNT);

current_af_pos = 0;

# all_features - matrix containing all extended features - rows are recognizables, columns are individual features
all_features = np.zeros( (max_datasize, asr.FEATURE_COUNT*(2*ctx_width+1) ) );
# all_classes - matrix containing 1-of-n NN woutpus - rows are recognizables, columns are individual outputs
all_classes  = np.zeros( (max_datasize, asr.CLASS_COUNT), dtype=np.int32 );



# ===============================================================================================
# Process subsequent utterances by extending feature vectors and selecting extended rows randomly
# ===============================================================================================
for i in range( len(features) ):
    # create extended feature vectors 
    process_utter( features[i], classes[i], ctx_width, fraction, max_datasize );
    if ( current_af_pos >= max_datasize ):
       break;

print "New feature vector len: ", str(len(all_features[0]));
    
# Now we have the array containing context-expanded features
print "Reshaping for Nielsen NNs";
print all_features.shape
print all_classes.shape

# ===============================================================================================
# Reshape according to third party NN code
# ===============================================================================================
if ( use_theano ):
    data = prepare_for_theano( all_features, all_classes );
else:    
    data = prepare_for_nn( all_features, all_classes );


# ===============================================================================================
# Now ready to store in a file
# ===============================================================================================    
print "Number of recognizables in dataset: " + str( len(data) );
hrs, mins, sec, dur = asr.frms2hrs( len(data) )
print "Data duration: (%s)" % ( dur );

print "Dumping data ...";
head, ext = os.path.splitext( out_file );
head = head + "." + "%02d" % ctx_width;
out_file = head + ext;
print "Set size: " + str( len(data) );
print "Feature veclen: " + str( len( data[0][0] ) );
print "Output veclen: " + str( len( data[0][1] ) );

data_ext = ( data, ctx_width*2 + 1)
f_out = open( out_file, "wb" );
pickle.dump( data_ext, f_out );
f_out.close();
print "Done";

# np.savez_compressed( out_file, a=all_features, b=classes );
