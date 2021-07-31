# Handwritten-digits-classification
Handwritten-digits-classification model using digits dataset.
Logistic Regression approach is used here.

Dataset used - digits dataset

* We can also access this data from the scikit-learn library. There are 506 samples and 13 feature variables in this dataset. The objective is to predict the value of prices of the house using the given features. 

## Description Of Dataset
The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents

## methodology
* To apply a classifier on this data, we need to flatten the images, turning each 2-D array of grayscale values from shape (8, 8) into shape (64,). Subsequently, the entire dataset will be of shape (n_samples, n_features), where n_samples is the number of images and n_features is the total number of pixels in each image.

* We can then split the data into train and test subsets and fit a support vector classifier on the train samples. The fitted classifier can subsequently be used to predict the value of the digit for the samples in the test subset.

### Tools and Languages:
<img align="left" alt="Google Colab" width="26px" src="colab.png" />
<img align="left" alt="Python" width="26px" src="python.png" />
<img align="left" alt="pandas" width="26px" height="34px" src="pandas.png" />
<img align="left" alt="numpy" width="36px" src="numpy.png" />
<img align="left" alt="matplotlib" width="26px" src="matplotlib.png" />
<img align="left" alt="Scikit_learn" width="26px" src="Scikit_learn.png" />
<br>

### Steps to follow
-Install the libraries<br>
-Download the code from this repository<br>
-Run the code using jupyter notebook<br>

### Developed by:
<a href="https://github.com/sambit221">Sambit Kumar Tripathy</a>