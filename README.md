# multi-letter-recognition
A deep learning model to recognize multi-letter sequences from images.


This model operates directly on the image pixels to identify multi-character sequences with varying lengths (based on [Goodfellow et al. 2013](https://arxiv.org/abs/1312.6082)) and avoids the traditional separated steps of localization, segmentation and recognition. A special output layer is built upon the deep network for this purpose. Here I assume a bounded sequence, say, the length of a sequence is at most 5. Then the output layer is made up of 6 softmax classifiers, one for the sequence length and the other five for each digit in the sequence.

To obtain the dataset, use [1notMNIST.ipynb] (basically taken from a [udacity course](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb)) to download and preprocess the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset which is designed to be similar to but harder than the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. Then my program [RecongnitionOfSequence.py] generates synthetic sequences of letters from the single-letter images in notMNIST. The sequence is limited to up to five letters. For sequences shorter than 5 letters, "blank" areas will be added to the end.

The neural network implemented in the program [RecongnitionOfSequence.py] consists of 3 convolutional layers, 2 fully connected layers and 1 output layer. Because of the limit of computational power and training set size, a smaller network might perform better. For example, my best test accuracy (nearly 90%) is achieved when I remove one fully connected layer. The whole training takes several days on my laptop.
