# CIFAR-10-Bayesian-Classifier
The CIFAR-10 Bayesian Classifier is an open-source project that implements a Bayesian classifier for the CIFAR-10 dataset. The project focuses on two main components: Naive Bayes classification and Multivariate Bayesian classification with different image shapes.

The Naive Bayes classifier is trained on the CIFAR-10 dataset, where it learns the class distributions based on color features. It calculates the mean and standard deviation for each class and uses these parameters to classify new test samples. The project provides a function to calculate the accuracy of the Naive Bayes classifier on the CIFAR-10 test set.

The Multivariate Bayesian classifier extends the classification approach to different image shapes, specifically 1x1, 2x2, 4x4, 8x8, and 16x16. It resizes the images accordingly and learns the class distributions using multivariate normal distributions. The classifier then predicts the labels for the test set images and calculates the accuracy for each image shape.

The project includes functions for data preprocessing, model training, and classification. Additionally, it provides visualization capabilities, such as displaying randomly selected CIFAR-10 images and plotting the accuracy of the Multivariate Bayesian classifier for different image shapes.

This project serves as a useful resource for understanding and implementing Bayesian classification techniques on the CIFAR-10 dataset. It offers flexibility in exploring different image shapes and provides insights into the performance of Bayesian classifiers for varying image resolutions.
