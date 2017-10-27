# nn_asr
ASR experimenting with Neural Networks using lynguistic resources for Polish language
=====================================================================================

This repository contains experimental code implementing CNNs for ASR tests based on CLARIN-PL resources. CLARIN-PL is Common Language Resources and Technology Infrastructure related project aimed on Polsih language )http://clarin-pl.eu/en/home-page/=.

The code is mainly based on lasagne tutorial examples. Modules stored in the repository implement the following functionalities:
- converting acoustic data in extended CLARIN-PL shape to the shape convinient to load with python/numpy code 
- extracting datasets consisting of specified fraction of bulk data and creating contextual feature vectors
- running CNN traininig and evaluation

Is is assumed that the input data come from acoustic corpus collected in the scope of CLARIN-PL. Orginal CLARIN-PL resources lack audio alignment to phonetic segments (phonemes). The phonetic transcription of utterances and their alignment to audio data are obtained using acoustic model trained on large amount of recordings in Polish.
