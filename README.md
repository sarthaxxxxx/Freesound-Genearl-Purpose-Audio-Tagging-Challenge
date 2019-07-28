# Freesound-General-Purpose-Audio-Tagging-Challenge
This is a simplified implementation of the Freesound General Purpose Audio Tagging Challenge. 

The AudioFiles file consists of 10 classes of musical instruments, each having 30 '.wav' files. 
The 'instruments.csv' file contains the names of the 'wav' files along with the class of musical instrument each audio belongs to.

The 'eda.py' file examines all the audio files and removes silence from each file. It also plots all the signlas, mfccs, ffts, and the filter bank for different characteristic parameters. All the new files are stored in a folder named 'clean'.

The 'model.py' file uses the audio files fom the 'clean' folder for feature extraction by considering the mfcc coefficients. By setting the mode of neural network, it is trained using CNN or LSTM architectures and the weights are saved.

The 'predict.py' does prediction/testing on the training audio files itself.

The predictions are stored in the 'Predictions.csv' file.


Just download the dataset and run all the '.py' files in the order given above.
