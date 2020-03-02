# SFWRTECH 4NN3 Course Project Convolutional Neural Network (CNN) for Face Detection

In this project, your primary task is to perform face detection on a small set of group photos. Face detection is a simple to define task – find regions in an image that contain pixels that represent a human face. As such, any region of the image can be classified as “face” or “non-face”. Therefore, the classification must be done on every possible region of the image at different scales (faces can be of different sizes).

We are going to use a convolutional neural network for the classification. The network is to be trained on the face images provided – a large set of 200x200 face images with the face centered in the image (i.e. the tip of the nose is in the center of the image). The face does not cover the entire 200x200 area of the training image so the area can be cropped if you wish.

Your CNN must contain at least 3 convolution layers. Each layer will contain 32 7x7 filters. Each layer must also contain a max pooling operation with a reduction in size by half for both width and height and a rectified linear unit for activation. The output of your last layer is flattened to provide the input layer of the fully connected multi-layer perceptron with 512 hidden neurons and one output neuron. You can use whichever of the available functions you would like for the loss function and whatever approach you would like to choose for the optimizer.
Note: To simplify matters a bit, do not use color information! Convert all images to greyscale for the purposes of both training and test.

Once the data set has been classified, your job is to compare the results from the classifiers that you chose.
The comparison (in general – details will follow below) must include the following analyses:

- Computational Times for both training and testing  
- A confusion matrix

## Dataset

- The data set contains more than 13,000 images of faces collected from    the web.  
- Each face has been labeled with the name of the person pictured.  
- 1680 of the people pictured have two or more distinct photos in the data set.  
- The only constraint on these faces is that they were detected by the Viola-Jones face detector.
- Link to dataset: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
