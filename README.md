# nn_asr
ASR experimenting with Neural Networks using lynguistic resources for Polish language
=====================================================================================

This repository contains experimental code implementing CNNs for ASR tests based on CLARIN-PL resources. CLARIN-PL is Common Language Resources and Technology Infrastructure related project aimed on Polsih language )http://clarin-pl.eu/en/home-page/=.

The code is mainly based on lasagne tutorial examples. Modules stored in the repository implement the following functionalities:
- converting acoustic data in extended CLARIN-PL shape to the shape convinient to load with python/numpy code 
- extracting datasets consisting of specified fraction of bulk data and creating contextual feature vectors
- running CNN traininig and evaluation

Is is assumed that the input data come from acoustic corpus collected in the scope of CLARIN-PL. Orginal CLARIN-PL resources lack audio alignment to phonetic segments (phonemes). The phonetic transcription of utterances and their alignment to audio data are obtained using acoustic model trained on large amount of recordings in Polish.

ASR experimenting with Neural Networks using lynguistic resources for Polish language
3
=====================================================================================
4
​
5
This repository contains experimental code implementing CNNs for ASR tests based on CLARIN-PL resources. CLARIN-PL is Common Language Resources and Technology Infrastructure related project aimed on Polsih language (http://clarin-pl.eu/en/home-page).
6
​
7
The code is mainly based on lasagne tutorial examples. Modules stored in the repository implement the following functionalities:
8
- converting acoustic data in extended CLARIN-PL shape to the shape convinient to load with python/numpy code 
9
- extracting datasets consisting of specified fraction of bulk data and creating contextual feature vectors
10
- running CNN traininig and accuracy evaluation
11
​
12
Is is assumed that the input data come from acoustic corpus collected in the scope of CLARIN-PL. Orginal CLARIN-PL resources lack audio alignment to phonetic segments (phonemes). The phonetic transcription of utterances and their alignment to audio data are obtained using acoustic model trained on large amount of recordings in Polish. 

Speech recording samples from CLARIN-PL corpus were segmented into 10ms frames. Each frame were assigned to the phone from teh sequence of phones in phonetic transcription of an utterance. Each frame was subject to feature extraction procedure. USED featires are Mel-scale spectral features obtained with 41 element filter bank. Additionally delta and delta-delta features were compyted resulting totally in 123 MFSC features per frame. For each frame a pair (feature vector, phoneme symbol) was created. Our aim is to create and train an NN which isbe fetched with features and produces on its output an estimation of phone probability provided given feature vector.

The subset of utterances from CLARIN-PL corpus was divided into training, validation and test subsets. For each subset the bulk dataset containing (features, phone) pairs were stored in bulk training/validation/testing datasets. Tools available in this repository make it possible to prepare particular training/validation/testing datasets for experiments by specifying:
- amount od data in datasets
- acoustic context for each frame.
13
Defining acoustic contex of a frame makes it possible to improve accuracy of phone recognition. The dataset extraction procedure combines several features from frames adjacent to the central one  ​
