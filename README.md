# catsVsDogs
Cats vs Dogs image classification with transfer learning from vgg16

A deep CNN implementation that modifies the last few layers of the pretrained vgg16 model and repurposes it to classify cat and dog images.

Depending on the selected subset of training and testing images, 97 to 99% accuracy is usually achieved. 

The code needs tensorflow, keras, numpy, pandas, matplotlib, sklearn, itertools and glob installed in the Python environment.

Download the dataset from: https://www.kaggle.com/competitions/dogs-vs-cats/data

Extract the contents of train folder to "./cats_vs_dogs/" folder under the project folder

During the first run the code will do a random sampling of the data form the train, test, validation sets. They will be reused in the subsequent runs. Number of images in each set can be modified as needed. If a new train, validation & test set is needed, delete the "test", "train", and "valid" folders under "./cats_vs_dogs/"

GPU parallelization is turned off, but it can be turned on by uncommenting the relevant line.
