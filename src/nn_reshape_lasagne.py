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
import math
from sys import getsizeof
import time

# ========================================================================================

def prepare_for_theano( all_features, all_classes ):
    """" 
       Prepares data to be read by Nielsen NNs.
    """
    print ("Not implemented");
    # Modify classes for exclusion
    sys.exit(1);

# ========================================================================================

def prepare_for_lasagne( all_features, all_classes ):
    """" 
       Prepares data to be read by Lasagne.
    """
    
    data = ( all_features, all_classes )
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
   global all_positions;   
   
   frm_cnt = len(features )
   rows_cnt = 2*ctx + 1;
   dst_f_len = asr.FEATURE_COUNT * ( rows_cnt );
   layer_size = asr.FEATURE_COUNT/3
   for i in range( frm_cnt ):
      # if ( np.random.uniform() > fraction ):
      #   continue;
      lin_pos = i - ctx;      
      for dst_pos in range( rows_cnt ):
         pos = max( lin_pos, 0 );
         pos = min( pos, frm_cnt - 1 );
         # all_features[ current_af_pos, 0, dst_pos ] = features[pos]         
         all_features[ current_af_pos, 0, dst_pos ] = features[pos][0:layer_size]         
         all_features[ current_af_pos, 1, dst_pos ] = features[pos][layer_size:2*layer_size]
         all_features[ current_af_pos, 2, dst_pos ] = features[pos][2*layer_size:3*layer_size]         
         lin_pos = lin_pos + 1;
         
      all_classes[ current_af_pos ] = np.argmax(classes[i]);
      all_positions[ current_af_pos ] = i
      current_af_pos = current_af_pos + 1;
      if ( current_af_pos >= max_datasize ):
         break;
      
   return current_af_pos;
      
# ===================================================================================
# ===================================================================================
# main (argv )
# ===================================================================================      
# ===================================================================================
def main():
    global current_af_pos;
    global all_features;
    global all_classes;
    global all_positions;
    
    if ( len( sys.argv ) < 4 ):
        print "Usage:";
        print "    python nn_reshape <in_file> <out_file> <context_width> [<fraction> [<max_segment_size_GB>]]\n"
        print "Context width - single side context - number of adjacent frames"
        print "Fraction - [0.0-1.0] determines fraction of data to be used"
        print "Max segment size - segment size limit in GB"        
        sys.exit( 1 );
    
    # ===================================================================================
    # Get and process parameters
    # ===================================================================================
    in_file = sys.argv[1];
    out_file = sys.argv[2];
    ctx_width = int( sys.argv[3] );

    fraction = 1.0;
    if ( len( sys.argv ) > 4 ):
       fraction = float( sys.argv[4] ) * 0.01;
    if ( fraction > 1.0 ):
       fraction = 1.0;       
       
    max_segment_size = 1e12
    if ( len( sys.argv ) > 5 ):
       max_segment_size = float( sys.argv[5] ) * 1e9;
    
    print "Max segment size [GB]: %s" % sys.argv[5];
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
    print "   Bulk frames count         : {0:,d}".format( frames_cnt );
    print "   Bulk utterances count     : {0:,d}".format( len(features) );

    hrs, mins, sec, dur = asr.frms2hrs( frames_cnt )
    max_datasize = min( int( frames_cnt * fraction ), frames_cnt );
    print "   Selected samples duration :",  dur;

    # Get actual features and frames count
    asr.FEATURE_COUNT = len( features[0][0] );
    asr.CLASS_COUNT = len( classes[0][0] );

    sample_size =  (asr.FEATURE_COUNT * 2 * (ctx_width+1) + asr.CLASS_COUNT ) * 4 
    segments_count = int(math.ceil( (max_datasize * sample_size) / float( max_segment_size)) )
    segment_max_datasize = int( max_datasize / segments_count )
    
    current_af_pos = 0;

    # all_features - matrix containing all extended features - rows are recognizables, columns are individual features
    # all_features = np.zeros( (max_datasize, 1, 2*ctx_width+1, asr.FEATURE_COUNT ) );
    all_features = np.zeros( (segment_max_datasize, 3, 2*ctx_width+1, asr.FEATURE_COUNT/3 ) );
    # all_classes - matrix containing 1-of-n NN woutpus - rows are recognizables, columns are individual outputs
    all_classes  = np.zeros( segment_max_datasize, dtype=np.int32 );
    all_positions  = np.zeros( segment_max_datasize, dtype=np.int32 );

    # ===============================================================================================
    # Process subsequent utterances by extending feature vectors and selecting extended rows randomly
    # ===============================================================================================
    print "Expected segments count %d" % ( segments_count, )
    seg_index = 0
    current_af_pos = 0;
    start_time = time.time()
    seg_start_time = time.time()
    utters_cnt = 0 
    for i in range( len(features) ):
        if ( np.random.uniform() <= fraction ):
            # create extended feature vectors 
            caf_start = current_af_pos
            process_utter( features[i], classes[i], ctx_width, fraction, segment_max_datasize );
            utters_cnt = utters_cnt + 1
            # print "utter %d/%d    cf_start: %d     cf_end: %d    utter_size: %d    incr: %d" % (i, len(features), caf_start, current_af_pos, len(features[i]), current_af_pos - caf_start  ) 
    
        if ( current_af_pos == 0 ):
           continue;
           
        if ( ( ( current_af_pos >= segment_max_datasize ) or ( i == len(features)-1 )) and ( current_af_pos != 0)):        
            all_features_dst = all_features[0:current_af_pos]
            all_classes_dst = all_classes[0:current_af_pos]
            all_positions_dst = all_positions[0:current_af_pos]     
            
            hrs, mins, sec, dur = asr.frms2hrs( len(all_features_dst) )

            head, ext = os.path.splitext( out_file );
            head = head + "." + "%02d" % ctx_width;
            out_file_seg = head + ( ".seg%03d"%seg_index ) + ext;  
            f_out = open( out_file_seg, "wb" );
            save_start_time = time.time()
            np.savez( f_out, a=all_features_dst, b=all_classes_dst, c=all_positions_dst );
            f_out.close()            
            
            print "Stored to segment file: %s" % (out_file, )
            print "   Stored frame count: %d/%d" % (current_af_pos, segment_max_datasize)
            print "   Stored data duration: (%s)" % ( dur );
            print "   Completed in  {:.3f}s   stored in {:.3f}s  ({:.3f})". \
                format( time.time() - seg_start_time, time.time() - save_start_time,
                        current_af_pos*sample_size*(1e-9)/(time.time() - save_start_time)   )
                        
            current_af_pos = 0;
            seg_index = seg_index + 1
            utters_cnt = 0
            seg_start_time = time.time();

    print "Terminated"

# ===================================================================================
# ===================================================================================
# main (argv )
# ===================================================================================      
# ===================================================================================
main()    