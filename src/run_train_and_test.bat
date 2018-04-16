@echo off
@echo Basic DNN training and testing (example with fixed train options)...
python %ASR_DATASET_ROOT%\src\nn_train_and_test.py %1 %2 %3 -e %4

goto :eof

rem ===============================================================================================================
:usage
@echo Usage:
@echo      %0 ^<feature_type^> ^<train_set^> ^<devel_set^> ^<test_set^> <^num_epochs^>
