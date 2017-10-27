# ========================================================================
# The program merges all cvn.nnin files located in the directory tree 
# specified by command line argument. The output file contains number of
# per frame features and number of phoneme clusters(classes for recognition)
# in the first line of the output file
# ========================================================================

import os
import sys
import glob
import asr
import numpy as np

if ( len( sys.argv ) < 3 ):
    print "Usage: ";
    print "   python merge_nnin <dir_tree> <output_file_name>";
    sys.exit( 1 );
    
path = sys.argv[1];
out_fname = sys.argv[2];

f_out = open( out_fname, "wt" );
is_first = True;

for dir, subdir, files in os.walk("."):
    for f in files:
        head, ext = os.path.splitext( f );
        if ( ext == ".nnin" ):
            fname = dir + "\\" + f;    
            # print "File: " + fname;
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
    