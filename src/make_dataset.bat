python %CLARIN_ROOT%\src\nn_reshape_new_opt.py train\train.nn.bin train.set.bin %1 %2
python %CLARIN_ROOT%\src\nn_reshape_new_opt.py test\valid.nn.bin valid.set.bin %1 
python %CLARIN_ROOT%\src\nn_reshape_new_opt.py core\core.nn.bin core.set.bin %1 

