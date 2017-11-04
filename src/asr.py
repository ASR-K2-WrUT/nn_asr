"""
   This file contains helper functions used in dataset preparation for ASR 
   experiments.
"""
import numpy as np
import sys
import re
import asr
import pickle

from tsp_solver import greedy
from random import random, seed
from time import time
from tsp_solver import greedy_numpy

# =============================================================================
# =============================================================================
# Global variables
# =============================================================================
# =============================================================================

FEATURE_COUNT = -1;
# Classes include unknown UNK phonem which is represented by the last elem 
# of teh output vector
CLASS_COUNT = -1;
# Constants definition
EOU_TAG = "****";
FREQ=16000
# number of frames per sec
FRAME_RATE = 100

# index in recognizable object arrays
current_af_pos = 0;

# phoneme lookup data for phoneme reordering
phone_lookup_fn = "phone_lookup.txt"


# =============================================================================
# =============================================================================
# Data loading functions
# =============================================================================
# =============================================================================  
          
def load_utters( in_file ):   
    """ Reads two numpy arrays containinig features and vectorized classses per frame
        Features are aggregated by utters, so both arrays are:
        features[ utters_cnt][ utter_len_in_frames ][ feature_cnt_per_frame ]
        classes array is organized in the same way
    """    
    data = np.load( in_file );
    return [ data['a'], data['b'] ];    
    
# =============================================================================
    
def prepare_for_lasagne( all_features, all_classes ):
    """" 
       Prepares data to be read by Lasagne.
    """
    
    data = ( all_features, all_classes )
    return ( data );   
    
# =============================================================================
    
def loadRawData( in_file, ctx_width=5, fraction=1.0 ):
    """ Loads raw data and reshapes it by merging sequences of features in subsequent
        frames. At the beginning and end of utterence frames are duplicated   
    """
    
    global current_af_pos;
    
    features, classes = load_utters( in_file );
    
    # Compute total number of frames in all utters
    frames_cnt = 0;
    for i in range( len(features) ): 
       frames_cnt = frames_cnt + len (features[i] )
    print "   Bulk frames count         : {0:,d}".format( frames_cnt );
    print "   Bulk utterances count     : {0:,d}".format( len(features) );

    max_datasize = min( int( frames_cnt * fraction ), frames_cnt );
    hrs, mins, sec, dur = asr.frms2hrs( max_datasize )
    print "   Selected samples duration :",  dur;

    # Get actual features and frames count
    asr.FEATURE_COUNT = len( features[0][0] );
    asr.CLASS_COUNT = len( classes[0][0] );

    current_af_pos = 0;

    # all_features - matrix containing all extended features - rows are recognizables, 
    #                columns are individual features
    all_features = np.zeros( (max_datasize, 3, 2*ctx_width+1, asr.FEATURE_COUNT/3 ) );
   
    # all_classes - matrix containing 1-of-n NN woutpus - rows are recognizables, 
    #               columns are individual outputs
    all_classes  = np.zeros( max_datasize, dtype=np.int32 );
   
    # =========================================================================
    # Process subsequent utterances by extending feature vectors and selecting 
    # extended rows randomly
    # =========================================================================
    for i in range( len(features) ):
        # create extended feature vectors 
        if ( np.random.uniform() > fraction ):
           continue;        
        process_utter( all_features, all_classes, features[i], classes[i], ctx_width, fraction, max_datasize );
        if ( current_af_pos >= max_datasize ):
           break;
       
    del features
    del classes
    
    return all_features, all_classes
          
          
# =============================================================================

def clipLine( line ):    
    line = line.rstrip();
    return line.lstrip();
    
# =============================================================================

def frms2hrs( frm_cnt ):    
    hr = 0;
    min = 0;
    sec = 0;
    
    frm_per_hour = int ( 3600 * FRAME_RATE );
    hrs = int(frm_cnt) / frm_per_hour;
    frm_cnt = frm_cnt - hrs * frm_per_hour;
    frm_per_min = int ( 60 * FRAME_RATE );
    mins = int(frm_cnt ) / frm_per_min;    
    frm_cnt = frm_cnt - mins * frm_per_min;
    sec = float(frm_cnt) / float(FRAME_RATE);
    dur_str = "%02dh:%02dm:%05.2fs" % ( hrs, mins, sec );
    return ( hrs, mins, sec, dur_str );

# =============================================================================

def load_nn_data( fname ):   
    f_in = open( fname, "rb" );
    data = pickle.load( f_in );
    f_in.close();
    return data;

# =============================================================================
    
def load_nn_data_out_plain( fname ):   
    f_in = open( fname, "rb" );
    data = pickle.load( f_in );
    mdata, ctx = data;
    f_in.close();
    for i in range( len(mdata) ):
        din, dout = mdata[i];
        dsout = devectorize( dout );
        mdata[i] = ( din, int(dsout) );
    return data;

# =============================================================================

def devectorize( data ):
    max_val = -10000.0;
    pos = -1;
    for i in range( len(data) ):
        if ( data[i] > max_val ):
            pos = i;
            max_val = data[i];
    return pos;

    
# =============================================================================
# =============================================================================
# Input/output data analysis functions
# =============================================================================
# =============================================================================

def get_stats( vect_data ):
    """
       Finds number of each class occurrence in the dataset. Input is the 
       array of NN outputs. Output is the vector where each element is the 
       number of occurrences of the correpsonding class.
    """
    cls_cnt = len( vect_data[0][1] );
    np.zeros( cls_cnt, dtype=np.int32 );
    outs = [ d[1].reshape( cls_cnt ) for d in vect_data ];
    stats = np.sum( outs, axis=0 );
    return stats;

# =============================================================================
    
def cls_stat( vect_data ):
    """
       Finds number and relative frequency of each class occurrence in the 
       dataset. Input is the array of NN outputs. Output is the vectors of 
       pairs of two values:
       - absolute class occurrence counts
       - relative frequency of a class 
    """
    cls_cnt = len( vect_data[0][1] );
    all_cls = [ d[1].reshape( cls_cnt ) for d in vect_data ];
    all_cls_np = np.array( all_cls );
    cls_stat = np.sum( all_cls_np, axis=0 );
    obj_cnt = len(vect_data);
    cls_stat_rel = cls_stat / float( obj_cnt );
    return zip( cls_stat, cls_stat_rel );

# =============================================================================
    
def cls_stat_save( stats, fname ):
    """
       Stores class histogram created by slc_stat function in a file.
    """
    f = open( fname, "wt ");
    sum_abs = 0; sum_rel = 0.0;
    for i in range( len(stats) ):       
       line = "%02d %6d  %6.3f \n" % (i, stats[i][0], stats[i][1] );
       sum_abs = sum_abs + stats[i][0];
       sum_rel = sum_rel + stats[i][1];       
       f.write( line );
    
    f.write( "\n   %6d  %6.3f \n" % ( sum_abs, sum_rel) );    
    f.close();   
    
# =============================================================================

def ft2histo( vect_data ):
    """
       Finds and prints ranges of feature values and their percentiles
    """
    ft_cnt = len( vect_data[0][0] );

    # find min/max of features
    all_ft = [ d[0].reshape( ft_cnt ) for d in vect_data ];
    all_ft_np = np.array( all_ft )
    mins = np.amin( all_ft_np, axis=0 );
    maxs = np.amax( all_ft_np, axis=0 );
    ranges = maxs - mins;
    print "Ranges: ";              
    print  ranges;
    perc15 = np.percentile( all_ft_np, 15.0, axis=0 );
    print( "Percentile 15: %4.2f:" % perc15 );              
    perc85 = np.percentile( all_ft_np, 85.0, axis=0 );    
    print( "Percentile 85: %4.2f:" % perc85 );              
    print  perc85 - perc15;
  
# =============================================================================
  
def  conf_mtr( results ):
    """
       Build statistics of output data, y_org, y_rec are list of pairs: 
       (y_reco, y_org) where y_reco/y_org are class indices
    """
    y_org  = np.array( [ org for (reco, org) in results ] );
    y_reco = np.array( [ reco for (reco, org) in results ] );
    
    out_size = len( y_org );
    if ( out_size != len( y_reco ) ):
        return None;
        
    # build histogram of true classes
    class_cnt = int(np.max( y_org ) + 1);
    histo = np.zeros( (class_cnt,) );
    for y in y_org:
       histo[y] = histo[y] + 1;
    histo = histo / np.sum( histo );
    
    # create confusion matrix    
    conf = np.zeros( (class_cnt, class_cnt ) );
    for y_t,y_r in zip( y_org, y_reco ):
        conf[y_t,y_r] = conf[y_t,y_r] + 1.0;
    
    conf_t = np.copy( conf );
    conf_t = np.transpose( conf_t);   
    
    for i in range( class_cnt ):
        conf[i] = conf[i] / np.sum( conf[i] );
    for i in range( class_cnt ):
        conf_t[i] = conf_t[i] / np.sum( conf_t[i] );

    distance = np.zeros( (class_cnt, class_cnt ) );
    for i in range( class_cnt ):
       for j in range( i, class_cnt ):
          if ( i == j ):
             distance[i,j] = 0.0
          else:
             distance[i,j] = 1.0 - (conf[i,j] + conf[j,i] ) / 2.0;
             distance[j,i] = 1.0 - (conf[i,j] + conf[j,i] ) / 2.0;             

    path = greedy_numpy.solve_tsp( distance, optim_steps = 20 )
    print path
    
    # get the vector of correct class recognitions            
    true_rec = np.zeros( (class_cnt,) );
    for i in range( len( conf ) ):
        true_rec[i] = conf[i,i];
    
    # Find two most frequent recognitions for each true class
    conf_cpy = np.copy( conf );
    max2 = []
    for i in range( class_cnt ):
       j = np.argmax( conf_cpy[i] );
       conf_cpy[i,j] = 0;
       j1 = np.argmax( conf_cpy[i] );
       conf_cpy[i,j1] = 0;
       j2 = np.argmax( conf_cpy[i] );       
       max2.append( (j,j1,j2) );
      
    return (histo, conf, conf_t, true_rec, max2, distance, path);  

# =============================================================================
    
def save_stats( stats, file, phone_file ):
    """
       Stores  statistics created by conf_mtr in a file
    """
    phones = []    
    with open( phone_file, "rt" ) as f_phn:
        for line in f_phn:
            line = asr.clipLine( line );
            phones.append( line );   
    
    print "Saving stats ...";
    histo, conf, conf_t, true_rec, max2, distance, path = stats;
    f = open( file, "wt" );
    f.write( "Class histogram: \n")
    for i in range( len( histo ) ):
        f.write( " %5.3f" % ( histo[i], ) );
        if ( i % 10 == 0 ):
            f.write( "\n" );
    f.write( "\n" );

    
    f.write( "Class histogram by phones: \n")
    for i in range( len( histo ) ):
        f.write( "%5s: %5.3f\n" % ( phones[i], histo[i] ) );
    f.write( "\n" );
    
    f.write( "Confusion matrix: (rows - expected;  cols - recognized;  c[row,col] = p(col is recog|row is expected) \n");    
    f.write( "Sum in rows ia 1.0 \n");      
    for i in range( len( conf ) ):
        for j in range( len( conf[i] ) ):           
            f.write( " %5.4f" % ( conf[i,j], ) );
        f.write( "\n" );
    f.write( "\n" );
    
    f.write( "Transposed confusion matrix: \n");    
    for i in range( len( conf_t ) ):
        for j in range( len( conf_t[i] ) ):           
            f.write( " %5.4f" % ( conf_t[i,j], ) );
        f.write( "\n" );
    f.write( "\n" );    
    
    f.write( "Distances (1-similarity): \n");    
    for i in range( len( distance ) ):
        for j in range( len( distance[i] ) ):           
            f.write( " %5.4f" % ( distance[i,j], ) );
        f.write( "\n" );
    f.write( "\n" );       
    
    f.write( "Correct recognition rates by classes: \n")
    for i in range( len( true_rec ) ):
        f.write( "%3d  %6.4f\n" % ( i, true_rec[i] ) );
    f.write( "\n" );

    f.write( "Most frequent recos by classes: \n")
    for i in range( len( max2 ) ):
        descr = phones[i] + " -> " + phones[max2[i][0]] + " " + phones[max2[i][1]] + " " + phones[max2[i][2]] 
        f.write( "%3d   %3d %3d %3d   %s\n" % ( i, max2[i][0], max2[i][1], max2[i][2], descr ) );
    f.write( "\n" );

    f.write( "Phones ordered by similarity: \n")
    for i in range( len(path) ):
        f.write( "%5s  %5.3f\n" % ( phones[path[i]], distance[ path[i], path[min(i+1, len(path)-1)] ] ) )
    f.write( "\n" );
   
    f.close();
    
    #  prapare phone reorder lookup
    lookup = np.zeros( len(path), dtype=np.int32 )
    if ( len(path) != len(phones) - 1 ):
       print( "Error in lookup size" );
       
    for i in range( len(path) ):
       lookup[ path[i] ] = i;
        
    f_out = open( phone_lookup_fn, "wb" );        
    np.savez( f_out, a=lookup );        
    f_out.close()    
        
    print "Saving stats completed";
       
    
# =============================================================================
# =============================================================================
# Auxiliary functions
# =============================================================================
# =============================================================================

def getFeatureCnt( fname ):
    """
        Detect actual feature per frame count from MFSC text file.
        It is assumed that each frame is represented by single line in mfsc file
    """ 
    f = open( fname, "rt" );
    line = f.readline();
    line = line.rstrip();
    line = line.lstrip();
    elems = re.split( "\ +", line );
    f.close();
    return len( elems );

# ================================================================

def getClassGrpCount( cls_grp_fname ):
    """
        Reads class cluster count (# of NN oputpus) from phoneme cluster 
        file. Each cluster is a nonempty line in cluster file. It is also assumed
        there there is UNK phoneme not directly represented in cluster file,
        therefore actual number of clusters is increased by 1.
    """
    cls_cnt = 0;
    with open( fname, "rt" ) as f:
        for line in f:           
            line = line.rstrip();
            line = line.lstrip();
            elems = re.split( "\ +", line );
            if ( len(elems) > 0 ):
               cls_cnt = cls_cnt + 1;
    return len( cls_cnt + 1 );

# ================================================================

def getClassCnt( fname ):
    """"
       Detects actual class count from clss file. The class count is 
       derived from the number of tokens in a single clss file line, where
       the number of tokens is equal to the number of class plus 2
    """
    f = open( fname, "rt" );
    line = f.readline();
    line = line.rstrip();
    line = line.lstrip();
    elems = re.split( "\ +", line );
 
    return len( elems) - 2 ;

# =============================================================================
    
def process_utter( all_features, all_classes, features, classes, ctx, fraction, max_datasize  ):
    """
       Combines per-frame feature vectors from frames close to the central one
       and builds contextual feature vector of a recognizable object (RO) corresponding
       to the central frame. The sequence of contextual features are stored in 
       all_features array. It also converts desired NN output vector into the class 
       index of the phoneme represented by a central frame or RO
    """
    global current_af_pos;
   
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
        current_af_pos = current_af_pos + 1;
        if ( current_af_pos >= max_datasize ):
            break;
      
    return current_af_pos;
        
    
