# Traffic Sign Recognition Project

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Dataset Exploration

#### Dataset Summary
Using the pickle library I load the dataset that have the following properties:

Number of training examples = 34799
Number of testing examples = 12630
Image shape = (32, 32, 3)
Number of classes = 43

'Number of classes' indicates the type of different signals are available into the dataset.  

#### Exploratory Visualization

As we can see from the following pciture

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Traffic-Sign-Classifier-Project/blob/master/sample_distribution.png  " ")

the training samples are not unifromly distributed among the 43 signal classes. 
In order to have a 
well trained NN it is important to train the model over the same number of samples for each class.
For that reason I augmented the data in order to have the same number of sample per class. 
The picture shows the new sample distribution among the signal classes.

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Traffic-Sign-Classifier-Project/blob/master/augmented_sample_distribution.png " ")

### Design and Test a Model Architecture

For what concerns the next steps and in the particular data preprocessing and augmentation I have been inspired by the
following [article](http://s3.amazonaws.com/academia.edu.documents/31151276/sermanet-ijcnn-11.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1491659143&Signature=nswY%2BFBhTX6HYB0xlxhQePo2zd0%3D&response-content-disposition=inline%3B%20filename%3DTraffic_Sign_Recognition_with_Multi-Scal.pdf). 

#### Preprocessing
I preprocessed the dataset by normalizing it and by greyscaling it. Such a procedure has several advantages such a reduction 
in the number of parameters we have to deal with.
The next two pictures show an original image (top) and the preprocessed one(bottom).

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Traffic-Sign-Classifier-Project/blob/master/signal_before_preprocessing.png " ")

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Traffic-Sign-Classifier-Project/blob/master/signal_after_preprocessing.png " ")

#### Model Architecture
I used a CNN based on an AlexNet architecture. The neural net composition is the follwoing:

**Layer 1 -> Convolutional:**  Input = 32x32x1 and Output = 28x28x6.

**Pooling ->**  Input = 28x28x6 and Output = 14x14x6.

**Layer 2 -> Convolutional:**  Output = 10x10x16.

**Pooling:**  Input = 10x10x16 and Output = 5x5x16.

**Flatten:** Input = 5x5x16 and Output = 400.

**Layer 3  -> Fully Connected:** Input = 400 and Output = 120.

**Relu activation.**

**Dropout:** prob = 0.5

**Layer 4  -> Fully Connected:**  Input = 120 and Output = 84.

**Relu Activation.** 

**Dropout:** prob = 0.5

**Layer 5  -> Fully Connected:** Input = 84 and Output = 10.
 
Both the pooling layers use a 2x2 filter, strides of 2 and a valid padding 

#### Model Training
In order to train the model I used 12 epochs, a bacth size of 64 and a learning rate with an initial value of 0.0005 that 
then decrease of 20% every four epochs.
All the weights have been iniziliazed following a gaussian distribution with mean equals to 0 and sigma equals to 0.1.
A reduced mean, cross entropy loss function was fed the logits from the last fully connected layer. This loss was
then minimized using the 'Adam' optimizer.

#### Solution Approach
I am kind of new to NN and for this reason I mainly based my project on my previous experience with the Udacity projects.
Since the project is focused on image classification I decided to use a convolutional neural network.
For the data preoprocessing I followed some suggetions from this [article](http://s3.amazonaws.com/academia.edu.documents/31151276/sermanet-ijcnn-11.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1491659143&Signature=nswY%2BFBhTX6HYB0xlxhQePo2zd0%3D&response-content-disposition=inline%3B%20filename%3DTraffic_Sign_Recognition_with_Multi-Scal.pdf). 
About the number and type of layers I mainly reused the experience from the LeNet-5 projects.
I then play around in order to find the values I used for the NN hyperparameter. 

### Test a Model on New Images

#### Acquiring New Images
The new images have been found on google image. 
One possible difficulty for the CNN is that in almost all the new 5 images there backgroudns on the image that can 
introduce in the sign image.

#### Performance on New Images
The cnn reached an accuracy of 80% on the new images. Even though such a value is lower than the 93% obtined during the model
evaluation, it is a very good result since it cannot be properly compared with the evaluation result due to the much lower
statistical significance od the new image sample.

#### Model Certainty - Softmax Probabilities
Finally I plot the 5 highest softmax probabilities obtained on the images downloaded from the web.

The following picture shows an example.






