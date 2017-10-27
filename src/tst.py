import network
import sys
import asr
import numpy as np

def compress( data ):   
    print "Compressing";
    for i in range( len(data) ):
        f,c = data[i];
        fo = 1.0 / ( 1.0 + np.exp( -1.0 * f ) );
        data[i] = (fo, c );

if ( len( sys.argv ) < 4 ):
   print "Usage python tst <train_file> <valid_file> <test_file> ";
   sys.exit(1);
   
print "Loading train data";
training_data, ctx1 = asr.load_nn_data( sys.argv[1] );
print "Loading verif data";
validation_data, ctx2 = asr.load_nn_data_out_plain( sys.argv[2] );
print "Loading test data";
test_data, ctx3 = asr.load_nn_data_out_plain( sys.argv[3] );

# asr.ft2histo( training_data );
# sys.exit(0);

# compress( training_data );
# compress( validation_data );
# compress( test_data );

if (( ctx1 != ctx2) or ( ctx1 != ctx3)):
   print "Data not equally shaped";
   sys.exit(1);

inputs_cnt = len( training_data[0][0] );
outputs_cnt = len( training_data[0][1] );    
print "Total context: " + str( ctx1 );
print "Inputs cnt: " + str( inputs_cnt );
print "Outputs cnt: " + str( outputs_cnt );

h,m,s,dur = asr.frms2hrs( len( training_data ) );
print "Training data size: " + str( len( training_data ) )+ " " + dur
h,m,s,dur = asr.frms2hrs( len( validation_data ) );
print "Valid data size: " + str( len( validation_data ) ) + " " + dur
h,m,s,dur = asr.frms2hrs( len( test_data ) );
print "Test data size: " + str( len( test_data ) )+ " " + dur
   
net = network.Network([inputs_cnt, 100, outputs_cnt] );
print "Network created";
net.SGD(training_data, 400, 100, 3.0, test_data = test_data);
   
# print "Avg res: {0}".format( sum_res / rept_cnt );

  
        
