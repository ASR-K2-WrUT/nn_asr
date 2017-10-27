import math
import asr
import sys
import re

def print_output( f_out, cls, cls_cnt ):
    for i in range( cls_cnt ):
        if ( i == cls ):
            f_out.write( " 1" );
        else:
            f_out.write( " 0" );

# ================================================================================
#  main( argv );
# ================================================================================
            
if (len (sys.argv) < 3 ):
    print "Usage:";
    print "     python timit2lab.py <phoneme_file_name> <phn_file_name> <clss_file_name>";
    sys.exit(1);
    
# Load phonemes file
phone_dict = dict( [] );
phone_file = sys.argv[1];

cls_cnt = 0;
with open( phone_file, "rt" ) as f_phn:
    for line in f_phn:
        line = asr.clipLine( line );
        group = re.split ( "\ +", line);
        if ( len(group) > 0 ):
            for phn in group:
                phone_dict[phn] = cls_cnt;
            cls_cnt = cls_cnt + 1;
print "Class cnt: " + str( cls_cnt );
print "Phonemes cnt: " + str( len ( phone_dict ) );

delta = 0.01; 
t0 = 0.0125;
f_out = open( sys.argv[3], "wt" );

with open( sys.argv[2], "rt" ) as f_in:
    for line in f_in:
        line = asr.clipLine( line );
        elems = re.split ( "\ +", line);
        sb = float( elems[0] );
        se = float( elems[1] );
        ts = sb / asr.FREQ;
        te = (se - 1.0) / asr.FREQ;
        # print "Ts: " + str(ts) + " si " + str( ts - t0 );
        si = math.ceil( (ts - t0)/delta );
        if ( si < 0):
            si = 0;
        ei = int( (te - t0)/delta )
        phn = elems[2];
        if ( ei >= si ):    
            if ( phn in phone_dict ):
                phn_ind = phone_dict[ phn ];      
            else:
                phn_ind = cls_cnt - 1;      
        
            f_out.write( "%d  %d    " % (si, ei) );
            print_output( f_out, phn_ind, cls_cnt );
            f_out.write("\n" );

f_out.write( "****\n" );
f_out.close();            

