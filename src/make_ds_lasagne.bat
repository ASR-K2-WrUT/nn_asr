python %CLARIN_ROOT%\src\nn_reshape_lasagne.py train\train.nn.bin train.set.%2.bin %1 %2
python %CLARIN_ROOT%\src\nn_reshape_lasagne.py test\valid.nn.bin valid.set.%2.bin %1 %3
python %CLARIN_ROOT%\src\nn_reshape_lasagne.py core\core.nn.bin core.set.%2.bin %1 %3

