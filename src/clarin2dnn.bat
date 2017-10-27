rem ==============================================================================================================
rem FOR CLARIN
rem ==============================================================================================================

rem @echo off

cd $CLARIN_ROOT%

rem ==============================================================================================================
rem Cleanup
for /R %%f in (*.clss) do del /Q %%f
for /R %%f in (*.mfsc) do del /Q %%f
for /R %%f in (*.nnin) do del /Q %%f


rem ==============================================================================================================
rem Build segmentation by frame numbers - clss files contain ranges of frames assigned to subsequent phones
rem and 1-of-n classes that will be used as NN output
for /R %%f in (*.out) do gawk -f d:/CLARIN\src\out2lab.gawk.txt %CLARIN_ROOT%\src\all_phones_grp.txt %%f >%%~pf%%~nf.clss

rem ==============================================================================================================
rem Create individual feature files 
for /R %%f in (*.mbin) do HList -r  %%f >%%~pf%%~nf.mfsc

rem ==============================================================================================================
rem Create feature file per speaker for the sake of normalization
for /R /D %%d in (*) do copy %%d\*.mfsc %%d\ALL._mfsc


rem ==============================================================================================================
rem Build merged feature-class files per speaker
for /R %%f in (ALL._mfsc) do python %CLARIN_ROOT%\src\build_fc.pl.py %%~pf %%~pfcvn.nnin

cd train
del /Q train.nn.txt
python %CLARIN_ROOT%\src\merge_nnin.py .\ train.nn.txt
rem for /R %%f in (*.nnin) do type %%f >>train.nn.txt
echo "Train samples cnt: "
wc -l train.nn.txt
python %CLARIN_ROOT%\src\nn_pack.py train.nn.txt train.nn.bin

cd ..\test
del /Q valid.nn.txt
python %CLARIN_ROOT%\src\merge_nnin.py .\ valid.nn.txt
rem for /R %%f in (*.nnin) do type %%f >>valid.nn.txt
echo "Test samples cnt: "
wc -l test.nn.txt
python %CLARIN_ROOT%\src\nn_pack.py valid.nn.txt valid.nn.bin

cd ..\core
del /Q core.nn.txt
python %CLARIN_ROOT%\src\merge_nnin.py .\ core.nn.txt
rem for /R %%f in (*.nnin) do type %%f >>core.nn.txt
echo "Core test samples cnt: "
wc -l core.nn.txt
python %CLARIN_ROOT%\src\nn_pack.py core.nn.txt core.nn.bin

cd ..

