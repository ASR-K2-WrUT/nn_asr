# nn_asr
ASR experimenting with Neural Networks using lynguistic resources for Polish language
=====================================================================================

This repository contains experimental code implementing CNNs for ASR tests based on CLARIN-PL resources. CLARIN-PL is Common Language Resources and Technology Infrastructure related project aimed on Polsih language (http://clarin-pl.eu/en/home-page). The aim of the project is to elaborate the set of programs and scripts that will meke it easier to convert data in the format applied in CLARIN-PL and TIMIT to the shape that can be used as traininig/development and test sets in NN training and evaluation. We assume that NN us used as phone probability estimator conditioned on observations. Observations are features extracted from acoustic data. The "master" branch contains tools elaborated for data in the format used in CLARIN-PL, while "TIMIT" branch contain their adaptation for TIMIT dataset.

The code is mainly based on lasagne tutorial examples. Modules stored in the repository implement the following functionalities:
- converting acoustic data in extended CLARIN-PL shape to the shape convinient to load with python/numpy code 
- extracting datasets consisting of specified fraction of bulk data and creating contextual feature vectors
- running CNN traininig and accuracy evaluation.

Is is assumed that the input data come from acoustic corpus collected in the scope of CLARIN-PL. Orginal CLARIN-PL resources lack audio alignment to phonetic segments (phonemes). The phonetic transcription of utterances and their alignment to audio data are obtained using acoustic model trained on large amount of recordings in Polish. 

Speech recording samples from CLARIN-PL corpus were segmented into 10ms frames. Each frame were assigned to the phone from teh sequence of phones in phonetic transcription of an utterance. Each frame was subject to feature extraction procedure. Used features are Mel-scale spectral features obtained with 41 element filter bank. Additionally, delta and delta-delta features were computed resulting totally in 123 MFSC features per frame. For each frame a pair (feature vector, phoneme symbol) was created. Our aim is to create and train an NN which is fetched with features and produces on its output an estimation of phone probability provided given feature vector.

The subset of utterances from CLARIN-PL corpus was divided into training, validation and test subsets. For each subset the bulk dataset containing (features, phone) pairs were stored in bulk training/validation/testing datasets. Tools available in this repository make it possible to prepare particular training/validation/testing datasets for experiments by specifying:
- amount od data in datasets
- acoustic context for each frame.

Defining acoustic contex of a frame makes it possible to improve accuracy of phone recognition. The dataset extraction procedure combines several features from frames adjacent to the central one and creates compound feature vector of a frame. In this way acoustic context of a frame can be taken into account wher recognizing a phone, the frame belongs to. The width of the context is defined by the number of frames on the single side (left or right) of the central frame - let us call it context radius. Typically the context radius used in ASR is between 1 to 10.

The Python readable data sets are prepared from raw recorded audio samples commplemented with its ortographic annotation. The datasets are stored as saved numpy arrays, so it can be read and used in any code written in Python.

The data shape of traininig/development/testing sets is organized as two arrays: for NN input and NN output. NN input 4D array is indexed by the quadruple of indices: (sample_index, slice_index, context_frame_index, feature_index). By "sample" we mean here the recognizable object, in case of phoneme recognition it corresponds to a single frame (extracted from 20ms segment of utterance audio recording). "slices" correspond to various feature types in the feature vector extracted from the single frame. We applied Mel-scale spectral coefficients (MFSC) acquired from the Mel-scale filter bank outputs, together with approximation of its ferst and second time derivatives (usually named: delta nad delta-delta). Filterbank outputs are stores in the first slice while deltas and delta-deltas constiture two remaining slices. \textit{context_frame_index} is the index of the frame within the assumed context. Taking context into account is achiewed by concatenating original frame feature vectors from the subsequence of frames actually surrounding the one being recognized.


Ver 101.d

