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

def prepare_for_nn( tr_d, fraction ):
    """" 
       Prepares data to be read by Nielsen NNs.
       It takes feature array containinig extended features (including context)
       and vectorized NN output and reshapes it according to format expected by NN.
       These two arrays are elements of tuple/pair passed via argument
    """
    # Modify classes for exclusion
    mask = np.zeros( (len(tr_d[0])), dtype=np.int8 );
    for i in range( len(tr_d[0]) ):
        if ( np.random.uniform() <= fraction ):
            mask[i] = 1;
    feature_veclen = len(tr_d[0][0]);
    out_veclen = len( tr_d[1][0]);
    inputs  = [ np.reshape(tr_d[0][x], (feature_veclen, 1)) for x in range(len(tr_d[0])) if mask[x] == 1]
    results = [ np.reshape(tr_d[1][y], (out_veclen,     1)) for y in range(len(tr_d[1])) if mask[y] == 1]
    data = zip( inputs, results)
    return ( data )      

# ========================================================================================

def prepare_for_theano( tr_d, fraction ):
    """" 
       Prepares data to be read by Nielsen NNs.
    """
    # Modify classes for exclusion
    mask = np.zeros( (len(tr_d[0])), dtype=np.int8 );
    for i in range( len(tr_d[0]) ):
        if ( np.random.uniform() <= fraction ):
            mask[i] = 1;
    feature_veclen = len(tr_d[0][0]);
    out_veclen = len( tr_d[1][0]);
    inputs  = [ tr_d[0][x]             for x in range(len(tr_d[0])) if mask[x] == 1]
    outputs = [ np.argmax(tr_d[1][y])  for y in range(len(tr_d[1])) if mask[y] == 1]

    np_inputs = np.array( inputs, dtype=np.float );    
    np_outputs = np.array( outputs, dtype=np.int32 );
    print "Theano shape inputs/outputs: ";
    print np_inputs.shape;
    print np_outputs.shape;
    
    data = ( np_inputs, np_outputs )
    return ( data );     
      
    
# ========================================================================================     
      
def load_utters( in_file ):   
    """ Reads two numpy arrays containinig features and vectorized classses
    """    
    data = np.load( in_file );
    return [ data['a'], data['b'] ];

# ========================================================================================  
    
def process_utter( features, classes, ctx  ):
   global current_af_pos;
   global all_features;
   # print features.shape;
   frm_cnt = len(features )
   dst_f_len = asr.FEATURE_COUNT * ( 2*ctx + 1 );
   feature_vec = np.empty( (2*ctx + 1 , asr.FEATURE_COUNT ) );
   # print all_features.shape;
   for i in range( frm_cnt ):
      dst_pos = 0;
      lin_pos = i - ctx;      
      for dst_pos in range( 2*ctx + 1 ):
         pos = max( lin_pos, 0 );
         pos = min( pos, frm_cnt - 1 );
         feature_vec[dst_pos] = features[pos];      
         dst_pos = dst_pos + 1;
         lin_pos = lin_pos + 1;
         
      lin_features = feature_vec.reshape ( (dst_f_len) );   
      # print "all_features " + str(current_af_pos);
      # print lin_features
      all_features[ current_af_pos ] = lin_features;
      current_af_pos = current_af_pos + 1;
      # np.savetxt( f_out, lin_features );
      #f_out.write( "   " );
      # np.savetxt( f_out, classes[i], fmt=" %3d" );    
      
# ===================================================================================
# main (argv )
# ===================================================================================      
if ( len( sys.argv ) < 4 ):
    print "Usage:";
    print "    python nn_reshape <in_file> <out_file> <context_width> [<fraction>]\n"
    print "Context width - single side context - number of adjacent frames"
    print "Fraction - [0.0-1.0] determines fraction of data to be used"
    sys.exit( 1 );

in_file = sys.argv[1];
out_file = sys.argv[2];
ctx_width = int( sys.argv[3] );

fraction = 1.0;
if ( len( sys.argv ) > 4 ):
   fraction = float( sys.argv[4] );

use_theano = False;
if ( len( sys.argv ) > 5 ):
    if( sys.argv[5] == "T" ):
        use_theano = True;
   
print "Fraction of " + str( fraction ) + " will be used"
print "Whole ctx width: " + str( 2*ctx_width+1);
   
features, classes = load_utters( in_file );
frames_cnt = 0;
for i in range( len(features) ): 
   frames_cnt = frames_cnt + len (features[i] )
print "Frames count: ", str( frames_cnt );
print "Utters count: ", str( len(features) );
hrs, mins, sec, dur = asr.frms2hrs( frames_cnt )
print "Total duration: ", str( frames_cnt * 0.01 ), " sec.  (" + dur + ")";

# Get actual features and frames count
asr.FEATURE_COUNT = len( features[0][0] );
asr.CLASS_COUNT = len( classes[0][0] );
print "Features count: %d" % (asr.FEATURE_COUNT);
print "Class count: %d" % (asr.CLASS_COUNT);

# Extend feature vectors to requested context
current_af_pos = 0;
all_features = np.zeros( (frames_cnt, asr.FEATURE_COUNT*(2*ctx_width+1) ) );

# print all_features.shape

for i in range( len(features) ):
    #print "Processing utter " + str(i);
    process_utter( features[i], classes[i], ctx_width );

print "New feature vector len: ", str(len(all_features[0]));
    
# Now we have the array containing context-expanded features
print "Reshaping for Nielsen NNs";
print all_features.shape
all_classes = np.concatenate(classes,axis=0)
print all_classes.shape
if ( use_theano ):
    data = prepare_for_theano( (all_features, all_classes), fraction );
else:    
    data = prepare_for_nn( (all_features, all_classes), fraction );

print "Number of recognizables in dataset: " + str( len(data) );
hrs, mins, sec, dur = asr.frms2hrs( len(data) )
print "Data duration: (%s)" % ( dur );

print "Dumping data ...";
head, ext = os.path.splitext( out_file );
head = head + "." + "%02d" % ctx_width;
out_file = head + ext;
data_ext = ( data, ctx_width*2 + 1)
f_out = open( out_file, "wb" );
pickle.dump( data_ext, f_out );
f_out.close();
print "Done";

# np.savez_compressed( out_file, a=all_features, b=classes );
