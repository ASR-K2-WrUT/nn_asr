#!/bin/bash 
#==============================================================================================================
# usage:
#     audio2dnn <feature_type>
# <feature_type> - mfsc (spectral)  or  mfcc (cepstral)
# Environment variable ASR_DATASET_ROOT must be set and point to the location of the folder
# containing training, development and testing datasets as subfolders. Bunary feature files
# must be located in these subfolders. All python and gawk programs should be located in 
# src subfolder of ASR_DATASET_ROOT.
#==============================================================================================================

if [ -n "$1" ]
then 
   FT_TYPE=$1
else
   echo "Usage:"
   echo "   audio2dnn <feature_type>"
   echo "<feature_type> - mfcc or mfsc"   
   exit 1
fi   

cd $ASR_DATASET_ROOT

# ==============================================================================================================
#Cleanup
echo "Removing files ..."
find . -name "*.clss" -exec rm {} \;
find . -name "*.${FT_TYPE}t" -exec rm {} \;
find . -name "*.nninout" -exec rm {} \;
find . -name "*ALL._${FT_TYPE}t" -exec rm {} \;

# ==============================================================================================================
# Build segmentation by frame subsequences - clss files contain ranges of frames assigned to subsequent phones
# and 1-of-n classes that can be used as desired NN output
echo "Preparing clss files"
for f in $(find . -name "*.out");
do
   dn=`dirname $f`
   fn=`basename $f .out`
   gawk -b -f $ASR_DATASET_ROOT/src/out2lab.gawk.txt $ASR_DATASET_ROOT/src/all_phones_grp.txt $f >$dn/$fn.clss
   echo "$f -> $dn/$fn.clss"
done   

# ==============================================================================================================
# Create individual feature files -it is assumed that binary feature files already exist
echo "Converting binary features (${FT_TYPE}) to text"
for f in $(find . -name "*.${FT_TYPE}");
do 
   dn=`dirname $f`
   fn=`basename $f .${FT_TYPE}`
   HList -r  $f >$dn/$fn.${FT_TYPE}t
   echo "$f -> $dn/$fn.${FT_TYPE}t"   
   touch $dn/ALL._${FT_TYPE}t   
done

# ==============================================================================================================
# Build merged feature-class files per speaker
echo "Merging inputs and outputs"
for f in $(find . -name "ALL._${FT_TYPE}t");
do 
   dn=`dirname $f`
   fn=`basename $f .${FT_TYPE}`
   python $ASR_DATASET_ROOT/src/ds_merge_f2c.py $dn $dn/${FT_TYPE}.nninout -t ${FT_TYPE}   
   echo "$dn/${FT_TYPE}.nninout" 
done

# ==============================================================================================================
# Create train, development and test datasets

cd train
rm -f train.${FT_TYPE}.txt
python $ASR_DATASET_ROOT/src/ds_merge_all.py . train.${FT_TYPE}.txt ${FT_TYPE}
echo "Train samples cnt: "
wc -l train.${FT_TYPE}.txt
python $ASR_DATASET_ROOT/src/ds_pack.py train.${FT_TYPE}.txt train.${FT_TYPE}.bin

cd ../devel
rm -f devel.${FT_TYPE}.txt
python $ASR_DATASET_ROOT/src/ds_merge_all.py . devel.${FT_TYPE}.txt ${FT_TYPE}
echo "Development samples cnt: "
wc -l devel.${FT_TYPE}.txt
python $ASR_DATASET_ROOT/src/ds_pack.py devel.${FT_TYPE}.txt devel.${FT_TYPE}.bin

cd ../test
rm -f test.${FT_TYPE}.txt
python $ASR_DATASET_ROOT/src/ds_merge_all.py . test.${FT_TYPE}.txt ${FT_TYPE}
echo "Core test samples cnt: "
wc -l test.${FT_TYPE}.txt
python $ASR_DATASET_ROOT/src/ds_pack.py test.${FT_TYPE}.txt test.${FT_TYPE}.bin

cd ..

exit 0

