@echo off
rem ==============================================================================================================
rem usage:
rem      audio2dnn <feature_type>
rem <feature_type> - mfsc (spectral)  or  mfcc (cepstral)
rem Environment variable ASR_DATASET_ROOT must be set and point to the location of the folder
rem containing training, development and testing datasets as subfolders. Binary feature files
rem must be located in these subfolders. All python and gawk programs should be located in 
rem src subfolder of ASR_DATASET_ROOT.
rem ==============================================================================================================

if [%1]==[] goto :usage

cd %ASR_DATASET_ROOT%

rem ==============================================================================================================
rem Cleanup
echo Deleting temporary files ...
for /R %%f in (*.clss) do del /Q %%f
for /R %%f in (*.%1t) do del /Q %%f
for /R %%f in (*.nninout) do del /Q %%f


rem ==============================================================================================================
rem Build segmentation by frame subsequences - clss files contain ranges of frames assigned to subsequent phones
rem and 1-of-n classes that can be used as desired NN output

echo Processing alignment files ...
for /R %%f in (*.out) do gawk -b -f %ASR_DATASET_ROOT%\src\out2lab.gawk.txt %ASR_DATASET_ROOT%\src\all_phones_grp.txt %%f >%%~pf%%~nf.clss

rem ==============================================================================================================
rem Create individual feature files -it is assumed that binary feature files already exist

echo Converting binary feature data to text ...
for /R %%f in (*.%1) do HList -r  %%f >%%~pf%%~nf.%1t

rem ==============================================================================================================
rem Create feature file per speaker for the sake of normalization
for /R /D %%d in (*) do copy %%d\*.%1t %%d\ALL._%1t

rem ==============================================================================================================
rem Build merged feature-class files per speaker

echo Merging output to input ...
for /R %%f in (ALL._%1t) do python %ASR_DATASET_ROOT%\src\ds_merge_f2c.py %%~pf %%~pf%1.nninout -t %1

rem ==============================================================================================================
rem Create train, development and test datasets

echo Preparing training data ...
cd train
del /Q train.%1.txt
python %ASR_DATASET_ROOT%\src\ds_merge_all.py .\ train.%1.txt %1
echo "Train samples cnt: "
wc -l train.%1.txt
python %ASR_DATASET_ROOT%\src\ds_pack.py train.%1.txt train.%1.bin

echo Preparing development data ...
cd ..\devel
del /Q devel.%1.txt
python %ASR_DATASET_ROOT%\src\ds_merge_all.py .\ devel.%1.txt %1
echo "Development samples cnt: "
wc -l devel.%1.txt
python %ASR_DATASET_ROOT%\src\ds_pack.py devel.%1.txt devel.%1.bin

echo Preparing test data ...
cd ..\test
del /Q test.%1.txt
python %ASR_DATASET_ROOT%\src\ds_merge_all.py .\ test.%1.txt %1
echo "Core test samples cnt: "
wc -l test.%1.txt
python %ASR_DATASET_ROOT%\src\ds_pack.py test.%1.txt test.%1.bin

cd ..

goto :eof

rem ===============================================================================================================
:usage
@echo Usage: %0 ^<feature_type^>
@echo ^<feature_type^> can be: mfcc or mfsc
@echo
@echo This script builds bulk recognizables packages from *.out and *mfsc/*.mscc files
@echo located in session folders. 
@echo
@echo Environment variable ASR_DATASET_ROOT must be set and point to the location of 
@echo the folder containing training, development and testing datasets as subfolders. 
@echo Binary feature files must be located in these subfolders. All python and gawk 
@echo programs should be located in src subfolder of ASR_DATASET_ROOT.
@echo 
@echo gawk, HList HCopy and python executables must be located in one of locations 
@echo defined in PATH environment variable.
