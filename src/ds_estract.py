"""
   This program prepares dataset used as training/development/test 
   dataset in speech recognition experiments with neural networks. 
   The shape of prepared data corresponds to the shape expected by NNs 
   created with Lasagne library (https://github.com/Lasagne/Lasagne) 
   but can be easily reshaped to fit other implementations of ASR-NN 
   training procedures.
   
   Typical ASR paradigm is based on phoneme sequence recognition using features
   extracted from short subsequent fragments (called frames) of utterance audio 
   signal. The typical role of NN in ASR lies in extimating phoneme conditional 
   probabilities for individual frames which are the used in certain sequential 
   model (e.g HMM or FST).

   The process of ASR dataset preparation is organized as follows. 
   The input data is the set of recorded relatively short (few seconds) 
   speech samples (utterances). For each utterance we have its phonetic 
   time-aligned transcription. 
   
   At the first stage of dataset building, each utterance undergoes feature 
   extraction with HCopy HTK module. It splits the sampled audio signal into 
   overlapping frames. Frame rate is 100Hz, but frame length is 20 ms. For 
   each frame the vector of MFSC or MFCC features is extracted (see HTK 
   documentation for details). The per-frame feature vector is complemented 
   with approximations of the first and second derivative approximation for 
   each individual MFCC/MFSC feature. These features are called delta and 
   delta-delta components. So per-frame feature vector consists of primary 
   MFSC/MFCC features and their delta and delta-delta complements.
   
   At the second stage, each frame is assigned a phone symbol of the phoneme 
   being uttered at the moment corresponding to a frame. For each utterance we 
   have its phonetic time-aligned transcription. Each frame in an utterance can 
   be assigned a phoneme symbol using time-aligned transcription. For each frame, 
   the pair is created consisting of per-frame feature vector and corresponding 
   phoneme symbol. In ASR we finally need to determine a phone using features 
   extracted from the audio stream. If ASR is based on HMM modelling instead 
   of expecting crisp phoneme recognition, we rather need extimation of each 
   phoneme probability given the vector of observations extracted from the frame. 
   (features, phoneme) pairs extracted from all recorded utterances in training, 
   development and testing sets are stored in raw "bulk" files containing all
   available data.
   
   Phoneme recognition accuracy may be higher if instead of using only per-frame 
   features also the frame context will be taken into account. It can be achieved 
   by using features not only extracted from a frame being recognized but also 
   from adjacent frames. While the aim is stil to recognize th phoneme of 
   the single frame, teh feature vector comprizes also per-frame features 
   of close neighbor frames. The width of the context is defined by context 
   radius "r". R usually comes from the set {0,...,15}. Number of concatenaded 
   per-frame feature vector is 2*r+1. To avoid confusion we will call the pair 
   consisting of extended feature vector and the phoneme of the central frame 
   "recognizable object".
   
   At the third stage of dataset preparation, recognizable object sets are created from 
   raw datasets created at the second stage. At this stage contexts are assembled by 
   concatenating per-frame feature vectors. In ASR experiments often we do not want 
   to use the whole available datasets but only randomly selected fraction of data 
   should be used as actual training dataset. At the third stage final datasets are 
   created by randomly selecting a subset of utterances and building recognizable 
   objects only from randomly selected utetrances.
   
   The program implemented in this source file provides the following functionalities:
   - selects specified fraction of utterance from the "bulk" data set,
   - created recognizable objects using specified context width,
   - stores the created dataset as numpy arrays convenient for further read 
     and application in NN training or performance evaluation.
   
   reads the array of individual utterances.
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