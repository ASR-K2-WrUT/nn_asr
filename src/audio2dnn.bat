rem ==============================================================================================================
rem usage:
rem      audio2dnn <feature_type>
rem <feature_type> - mfsc (spectral)  or  mfcc (cepstral)
rem Environment variable ASR_DATASET_ROOT must be set and point to the location of the folder
rem containing training, development and testing datasets as subfolders. Bunary feature files
rem must be located in these subfolders. All python and gawk programs should be located in 
rem src subfolder of ASR_DATASET_ROOT.
rem ==============================================================================================================

rem @echo off

cd %ASR_DATASET_ROOT%

rem ==============================================================================================================
rem Cleanup
rem for /R %%f in (*.clss) do del /Q %%f
rem for /R %%f in (*.%1t) do del /Q %%f
for /R %%f in (*.nninout) do del /Q %%f


rem ==============================================================================================================
rem Build segmentation by frame numbers - clss files contain ranges of frames assigned to subsequent phones
rem and 1-of-n classes that can be used as desired NN output

for /R %%f in (*.out) do gawk -f %ASR_DATASET_ROOT%\src\out2lab.gawk.txt %ASR_DATASET_ROOT%\src\all_phones_grp.txt %%f >%%~pf%%~nf.clss

rem ==============================================================================================================
rem Create individual feature files -it is assumed that binary feature files already exist
for /R %%f in (*.%1) do HList -r  %%f >%%~pf%%~nf.%1t

rem ==============================================================================================================
rem Create feature file per speaker for the sake of normalization
for /R /D %%d in (*) do copy %%d\*.%1t %%d\ALL._%1t

rem ==============================================================================================================
rem Build merged feature-class files per speaker
for /R %%f in (ALL._mfct) do python %ASR_DATASET_ROOT%\src\build_fc.py %%~pf %%~pf%1.nninout %1t

rem ==============================================================================================================
rem Create train, development and test datasets

cd train
del /Q train.nn.txt
python %ASR_DATASET_ROOT%\src\merge_nnin.py .\ train.%1.txt
echo "Train samples cnt: "
wc -l train.nn.txt
python %ASR_DATASET_ROOT%\src\dataset_pack.py train.%1.txt train.%1.bin

cd ..\devel
del /Q devel.%1.txt
python %ASR_DATASET_ROOT%\src\merge_nnin.py .\ devel.%1.txt
echo "Development samples cnt: "
wc -l devel.nn.txt
python %ASR_DATASET_ROOT%\src\dataset_pack.py devel.%1.txt devel.%1.bin

cd ..\test
del /Q test.%1.txt
python %ASR_DATASET_ROOT%\src\merge_nnin.py .\ test.%1.txt
echo "Core test samples cnt: "
wc -l test.nn.txt
python %ASR_DATASET_ROOT%\src\dataset_pack.py test.%1.txt test.%1.bin

cd ..

