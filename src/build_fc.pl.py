# This script is called "per speaker".It processes and merges all data 
# extracted from utterances of the same speaker. So CVN normalization
# is being done on all feature vectors taken from all mfsc files 
# prepared individually for recorded utterances

import os
import sys
import glob
import asr
import numpy as np

def writef( f_out, val ):
    str = " %18.11e" % val;
    f_out.write( str );

def writei( f_out, val ):
    str = " %d" % val;
    f_out.write( str );

def writeln( f_out ):
    str = "\n" 
    f_out.write( str );
    


if (len(sys.argv) != 3):
    print "Usage: build_fc.py <path> <out_fname>";
    sys.exit(1);
    
path = sys.argv[1];
out_fname = sys.argv[2];
ft_type = "mfsc"
if (len(argv> > 3 ):
   ft_type = sys.argv[3];
   
fname_by_type = path + "\\*." + ft_type
all_mfsc = glob.glob( fname_by_type );
all_class = glob.glob( path + "\\*.clss" );

if ( len(all_mfsc) != len(all_class) ):
    print "_mfsc and clss files do not match mfsc %d     clss %d" % ( len(all_mfsc) , len(all_class) );
    sys.exit();
    
if ( len( all_mfsc) == 0 ):
    sys.exit(0);

# ================================================================
# Detect actual feature count per frame, it is assumed that each 
# frame is represented by single line in mfsc file
# ================================================================
asr.FEATURE_COUNT = asr.getFeatureCnt( all_mfsc[0] );
asr.CLASS_COUNT = asr.getClassCnt( all_class[0] );

    
# ================================================================    
# Normalize features and merge classes and features   
# ================================================================
f_out = open( out_fname, "wt" );
# print "Counts %d %d" % (int(asr.FEATURE_COUNT), int(asr.CLASS_COUNT) )
f_out.write( "%d %d\n" % (int(asr.FEATURE_COUNT), int(asr.CLASS_COUNT) ) )
  
curr_row = 0;   
# t_adata = np.empty( (0, asr.FEATURE_COUNT) );
# print t_adata.shape;

for f_ft, f_cls in zip( all_mfsc, all_class ):
    t_in = np.fromfile( f_ft , sep=" " );
    t_in = t_in.reshape( (-1, asr.FEATURE_COUNT ) );    
    ft_rc, ft_cc  = t_in.shape;

    t_t = t_in.transpose();
    f_mean = [];
    f_std = [];
    for i in range( asr.FEATURE_COUNT ):
       f_mean.append( t_t[i].mean() );
       f_std.append( t_t[i].std() );
    
    # Normalize features
    for r in range( ft_rc ):
       for c in range( ft_cc ):
           t_in[r,c] = (t_in[r,c] - f_mean[c]) / f_std[c];
   
    # Read segmentation by phones      
    # print "Processing class file " + f_cls + "  " + str(rc);
    t_seg = np.fromfile( f_cls , dtype=np.int32, sep=" " );
    # here CLASS_COUNT includes also "sp" and all possible unknown class representing all phones excluded from recognition
    t_seg_mtx = t_seg.reshape( (-1, asr.CLASS_COUNT + 2 ) );
    seg_cnt, c_cnt = t_seg_mtx.shape;
    
    # create fragment of output file
    for seg in range( seg_cnt ):
        if ( t_seg_mtx[seg,0] >= ft_rc ):
            t_seg_mtx[seg,0] = ft_rc - 1;
        if ( t_seg_mtx[seg,1] >= ft_rc ):
            t_seg_mtx[seg,1] = ft_rc - 1;            
        for frm in range( t_seg_mtx[seg,0], t_seg_mtx[seg,1] + 1):
            # t_adata = np.append( t_adata, np.zeros((1, asr.FEATURE_COUNT)),  axis=0 );
            for feature in range(asr.FEATURE_COUNT):
                writef( f_out, t_in[frm, feature] );
                # t_adata[curr_row, feature] = t_in[frm, feature];                
            for out_ind in range(asr.CLASS_COUNT):
                writei( f_out, t_seg_mtx[seg, out_ind + 2] );
            writeln( f_out );
            curr_row = curr_row + 1;
    
    # Put "end-of-utterance" tag
    f_out.write( "**** " + f_ft + "\n" );

    
# t_adata_t = t_adata.transpose();
# for f in range( FEATURE_COUNT ):
#     print "Tested: " + str( t_adata_t[f].mean() ) + "\n";
    
f_out.close();
