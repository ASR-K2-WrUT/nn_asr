import os
import re
import sys

if ( len(sys.argv) < 2 ):
   print "Usage: python create_phone_set.py <output_file_name> "
   sys.exit( 1 );

f_out = sys.argv[1];
   
# browse all phn files
phones = set( [] );
for dir, subdir, files in os.walk("."):
    for f in files:
        head, ext = os.path.splitext( f );
        if ( ext == ".phn" ):
           fname = dir + "\\" + f;
           # print "File: " + fname + "   " + dir + "   " + f 
           
           # Get all phones
           f = open( fname, "rt");
           
           for line in f:
              line = line.rstrip();
              line = line.lstrip();
              
              elems = re.split( "\ +", line );
              # in phn TIMIT file the phone symbol is exactly on third position in each line
              phones.add( elems[2] );
           f.close();
           
print "Phone cnt: " + str( len(phones) );

f = open( f_out, "wt" );
for phn in phones:
   f.write( phn + "\n");
f.close();