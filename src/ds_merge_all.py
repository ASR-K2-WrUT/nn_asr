"""
    The program merges all cvn.nninout files located in the directory tree 
    specified by command line argument. The output file contains in the first 
    line number of per frame features and number of phoneme clusters (classes 
    for recognition) which is equal to the number of NN outputs.
    The body of the file is the simple concatenation of files containing 
    "per frame" (features, nn_outputs) pairs included in subfolders. It is 
    assumed that subfolders contain data extracted from utterances comming
    form individual speakers, i.e. utetrances in the single subfolder come
    from the same speaker.
    
    In assumed usage scenario, utterances constituting training, development 
    and testing sets should be gathered in three subfolder trees and 
    ds_merge_all.py program should be invoked in roots of these three folder 
    trees.
"""

import os
import sys

if ( len( sys.argv ) < 4 ):
    print "Usage: ";
    print "   python ds_merge_all.py <dir_tree> <output_file_name> <feature_type>";
    print "<feature_type> - mfcc or mfsc"
    sys.exit( 1 );
    
path = sys.argv[1];
out_fname = sys.argv[2];
feature_type = sys.argv[3]

f_out = open( out_fname, "wt" );
is_first = True;

for dir, subdir, files in os.walk("."):
    for f in files:
        head, ext = os.path.splitext( f );        
        if (( ext == (".nninout") ) and (head == feature_type )):
            fname = dir + os.sep + f;    
            f = open( fname, "rt" );
            line = f.readline();
            if ( is_first ):
                f_out.write( line )
                is_first = False;
            is_eof = False;
            while ( not is_eof ):
                line = f.readline();
                if (len( line) > 0 ):
                    f_out.write( line );
                else:
                    is_eof = True;
            f.close();

f_out.close()
    