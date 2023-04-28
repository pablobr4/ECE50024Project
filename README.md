# ECE50024 Project - Pablo Bedialauneta

This repository holds the code for the ECE50024 project by Pablo Bedialauneta.

First, make sure to install the following dependencies for Python 3.X:

- torch, torchvision and torchaudio.
- Pillow
- Numpy
- Matplotlib
- Glob

In order to run the files, it is important to first set up the images directories. There are two options for this.

1) Download the images.zip file and extract it in the folder in which the code is going to run. This way, the train.py file (and all other .py files) and the "images" folder should be in the same directory. Inside the images folder, there should be two folders, "train" and "test", each of which has many classes with several images per class.

2) Create your own extended MNIST dataset with the create_data.py script (internet connection required). For this to work, just create a folder named "images" in the same directory as create_data.py, and inside the "images" folder create another two folders: "train" and "test". Then, run the create_data.py script, and it will populate these folders.

After all dependencies are installed and the images directory is correctly created, download all .py files into the same directory as the "images" folder.
To train the metalearner, just open the train.py file, scroll down to the "USER DEFINED PARAMETERS" comment and fill them with the desired parameters. By default, the parameters are set to 10 classes per episode, 5-shot classification with 15 images for metatesting, to be trained for 1500 episodes and 5 metaepochs per episode.

When running the train.py file, the running loss is printed every 10 episodes. After training is finished, the metalearner model is saved in a file named "metalearner.pth" and the state dictionary is saved as "metalearner_dict.pth". The running loss plot is saved as "loss.png" and the running loss data as "loss_arr.pth".

To compute the accuracy of the learner when trained with the metalearner LSTM, open the test.py file, scroll down to the "USER DEFINED PARAMETERS" and fill them. The "num_classes_episode" parameter should be the same as the one used for training the metalearner. Make sure that the path to the metalearner model to be loaded is correct. After running this script, the accuracy is printed as a percentage of the total images correctly classified. This percentage is also saved in the file "accuracy.pth".

In order to compare the accuracy to the baseline implementation, you can run the "train_baseline.py" file. Again, scroll down to the "USER DEFINED PARAMETERS" and fill them. After running this script, the accuracy of the baseline learner is printed and saved in the "baseline_acc.pth" file.

In the "saved_models" folder of this repo you can find the already trained models for 1-shot up to 10-shot classification.

In the "results" folder you can find the results presented in the paper for the accuracies and training losses.
