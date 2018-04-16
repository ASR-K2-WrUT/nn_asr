#!/bin/bash 
if [ -z "$3" ]
then 
   echo "     $0 <feature_type> <contex_width> <train_data_raction> <test_data_fraction>"
   echo "<feature_type>       - mfsc or mfcc"
   echo "<context_width>      - single side context width in frames"
   echo "<train_data_raction> - fraction of data in input bulk dataset to be used"
   echo "                       for training (0-100)"
   echo "<test_data_raction>  - fraction of data in input bulk dataset to be used"
   echo "                       for development and testing (0-100)"
   exit 1
else
   echo OK
fi
python $ASR_DATASET_ROOT/src/ds_extract.py train/train.$1.bin train.set.$3.bin $2 $3
python $ASR_DATASET_ROOT/src/ds_extract.py devel/devel.$1.bin devel.set.$4.bin $2 $4
python $ASR_DATASET_ROOT/src/ds_extract.py test/test.$1.bin test.set.$4.bin $2 $4

