import network2
import sys
import asr
import numpy as np

fname_core = "d:\\NN_clarin.size1"

def compress( data ):   
    print "Compressing";
    for i in range( len(data) ):
        f,c = data[i];
        # fo = 1.0 / ( 1.0 + np.exp( -1.0 * f ) );
        fo = 0.3 * f;
        data[i] = (fo, c );       
       
if ( len( sys.argv ) < 4 ):
   print "Usage python tst <train_file> <valid_file> <test_file> ";
   sys.exit(1);
   
print "Reporting to files: " + fname_core
print "Loading train data";
training_data, ctx1 = asr.load_nn_data( sys.argv[1] );
cls_stat = asr.cls_stat( training_data );
asr.cls_stat_save( cls_stat, sys.argv[1] + ".stat.txt" );

print "Loading verif data";
validation_data, ctx2 = asr.load_nn_data_out_plain( sys.argv[2] );
print "Loading test data";
test_data, ctx3 = asr.load_nn_data_out_plain( sys.argv[3] );

"""
compress( training_data );
compress( validation_data );
compress( test_data );
"""

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
   
def NN_Test_single( prm ):
    best_result = [];
    ITER_CNT = 5;
    rep_fname = fname_core + ".final.txt";
    for iter in range( ITER_CNT ):
        # """
        net = network2.Network([inputs_cnt, prm["latent_nn_cnt"], outputs_cnt], rep_file=fname_core+".stats" );
        # net.large_weight_initializer()
        print "Network created";
        eval_cost, eval_acc, train_cost, train_acc = \
           net.SGD( training_data, 150, prm["batch_size"], prm["eta"], evaluation_data=test_data, \
                    lmbda=prm["lmbda"], \
                    monitor_evaluation_accuracy=True, monitor_training_accuracy=True );
        best_iter_ind = np.argmax( eval_acc );
        eval_acc.sort();
        best_result.append( eval_acc[-1] );
        best_result.sort();
        f = open( rep_fname, "at+" );
        f.write( str( (prm, [ "%6.4f" % v for v in best_result]) ) + "  best_it:" + str(best_iter_ind) + "\n" );
        f.close();

    return best_result[-1];

#NN_Test_single( { "latent_nn_cnt":30,  "batch_size":40, "eta":0.1, "lmbda":0.4 } );  
# NN_Test_single( { "latent_nn_cnt":100,  "batch_size":40, "eta":0.05, "lmbda":0.4 } );  
#NN_Test_single( { "latent_nn_cnt":200,  "batch_size":40, "eta":0.1, "lmbda":0.4 } );
NN_Test_single( { "latent_nn_cnt":350,  "batch_size":40, "eta":0.05, "lmbda":0.4 } );
#NN_Test_single( { "latent_nn_cnt":400,  "batch_size":40, "eta":0.1, "lmbda":0.4 } );
#NN_Test_single( { "latent_nn_cnt":500,  "batch_size":40, "eta":0.1, "lmbda":0.4 } );



  
        
