@echo off
if [%3]==[] goto :usage
@echo Extraction may take several minutes ...
python %ASR_DATASET_ROOT%\src\ds_extract.py train\train.%1.bin train.%1.%3.bin %2 %3
@echo Train data saved in train.%1.%4.%3.bin file
python %ASR_DATASET_ROOT%\src\ds_extract.py devel\devel.%1.bin devel.%1.%4.bin %2 %4
@echo Validation data saved in devel.%1.%4.%3.bin file
python %ASR_DATASET_ROOT%\src\ds_extract.py test\test.%1.bin test.%1.%4.bin %2 %4
@echo Test data saved in test.%1.%4.%3.bin file

goto :eof

rem ===============================================================================================================
:usage
@echo Usage:
@echo Usage:
@echo      %0 ^<feature_type^> ^<contex_width^> ^<train_data_traction^> ^<test_data_fraction^>
@echo ^<feature_type^>       - mfsc or mfcc
@echo ^<context_width^>      - single side context width in frames
@echo ^<train_data_raction^> - fraction of data in input bulk dataset to be used 
@echo                        for training (0-100)
@echo ^<test_data_raction^>  - fraction of data in input bulk dataset to be used 
@echo                        for development and testing (0-100)
