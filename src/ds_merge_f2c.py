"""
    This script is called individually for each speaker folder, where all
    utterances from the same speaker are stored. It processes and merges 
    all data extracted from utterances of the same speaker. So CVN normalization
    is being done on all feature vectors taken from all mfsc files 
    prepared individually for recorded utterances. 
    
    The proram reads files containing features extracted from individual 
    utterance audio files (*.mfsc or *.mfcc), and corresponding file containing 
    expected NN output for subsequences of frames (*.clss). mfsc/mfcc files 
    contain features extracted from audio signal for subsequent 20ms frames. 
    clss files contain phoneme allignment information which makes it possible 
    to assign each frame to a phone.
    
    The program merges features and classes information so that pairs (features, class)    
    can be used as inputs and outputs of NN in traininig and testing phases.
    
    The program creates the text file containing pairs (features, classes) for each 
    frame. Classes are represented by 0-1 vector having 1 on the single position 
    corresponding to a phoneme the frame belongs to and 0s on all other positions. 
    The features are CVN normalized on per-utterance basis. Blocks of lines 
    corrsponding to various utterances are separated by a tag line containing **** 
    sequence.
    
"""
import os
import sys
import glob
import asr
import numpy as np
import argparse

# =============================================================================
# Helper functions
# =============================================================================

def writef( f_out, val ):
    str = " %18.11e" % val;
    f_out.write( str );

def writei( f_out, val ):
    str = " %d" % val;
    f_out.write( str );

def writeln( f_out ):
    str = "\n" 
    f_out.write( str );
    

# =============================================================================
# main function
# =============================================================================
       
def main( params ):    
      
    path, out_fname, ft_type = params
    
    # Create lists of mfsc/mfcc and clss files
    fname_by_type = path + os.sep + "*." + ft_type + "t"
    all_mfsc = glob.glob( fname_by_type )
    all_class = glob.glob( path + os.sep + "*.clss" )
    all_mfsc.sort()
    all_class.sort()
    
    if ( len(all_mfsc) != len(all_class) ):
        print "feature and clss files do not match feature %d     clss %d" % ( len(all_mfsc) , len(all_class) );
        sys.exit();
        
    if ( len( all_mfsc) == 0 ):
        print "No files detected"
        sys.exit(0);

    # ================================================================
    # Detect actual feature count per frame, it is assumed that each 
    # frame is represented by single line in mfsc file
    # ================================================================
    asr.FEATURE_COUNT = asr.getFeatureCnt( all_mfsc[0] )
    asr.CLASS_COUNT = asr.getClassCnt( all_class[0] )

        
    # ================================================================    
    # Normalize features and merge classes and features   
    # ================================================================
    f_out = open( out_fname, "wt" )
    f_out.write( "%d %d\n" % (int(asr.FEATURE_COUNT), int(asr.CLASS_COUNT) ) )
      
    curr_row = 0   

    for f_ft, f_cls in zip( all_mfsc, all_class ):
        t_in = np.fromfile( f_ft , sep=" " );
        t_in = t_in.reshape( (-1, asr.FEATURE_COUNT ) );    
        ft_rc, ft_cc  = t_in.shape;

        # Create the features matrix where rows correspond to features 
        # and columns correspond to frames in a single utterance
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
        t_seg = np.fromfile( f_cls , dtype=np.int32, sep=" " );
        # Here CLASS_COUNT includes also "sp" and all possible unknown class representing all phones excluded from recognition
        t_seg_mtx = t_seg.reshape( (-1, asr.CLASS_COUNT + 2 ) );
        seg_cnt, c_cnt = t_seg_mtx.shape;
        
        # Create fragment of output file
        for seg in range( seg_cnt ):
            if ( t_seg_mtx[seg,0] >= ft_rc ):
                t_seg_mtx[seg,0] = ft_rc - 1;
            if ( t_seg_mtx[seg,1] >= ft_rc ):
                t_seg_mtx[seg,1] = ft_rc - 1;            
            for frm in range( t_seg_mtx[seg,0], t_seg_mtx[seg,1] + 1):
                for feature in range(asr.FEATURE_COUNT):
                    writef( f_out, t_in[frm, feature] );
                for out_ind in range(asr.CLASS_COUNT):
                    writei( f_out, t_seg_mtx[seg, out_ind + 2] );
                writeln( f_out );
                curr_row = curr_row + 1;
        
        # Put "end-of-utterance" tag
        f_out.write( "**** " + f_ft + "\n" );
        
    f_out.close();    

# =============================================================================    
# Program
# =============================================================================    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog = "ds_merge_f2c.pl",
                description = "This program processes and merges all data  extracted from utterances " + 
                              "of the same speaker.\nShould be called individually foe each speaker folder.",
                usage='\n    %(prog)s <path> <out_fname> [options]'
             )
    parser.add_argument( '-t', type=str, dest='type', default='mfsc', 
                         help=' - type features:mfcc (ceptral) or mfsc (spectral); default: mfsc ')
    
    args = parser.parse_args( sys.argv[3:])

    if ( len(sys.argv) < 3 ):
       parser.print_help()
       sys.exit(1)  
     
    params = ( sys.argv[1], sys.argv[2], args.type )
    main( params )