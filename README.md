## COVID19_Classification
This repository contains data for COVID19 and a Transfer Learning Based model for classification

There are two types of classification that were done.

* A four class classification among X-Ray Images. The classes were Nomral X Rays, COVID 19 X Rays, Viral Pneumonia, Bacterial Pneumonia etc.
* A two class classification on COVID 19 X rays and Normal X Rays 

The COVID-19 X-ray image dataset was curated by Dr. Joseph Cohen, a postdoctoral fellow at the University of Montreal. If you want to contribute in his work, please visit his [Github Repository](https://github.com/ieee8023/covid-chestxray-dataset)

I used RESNET-50 based transfer learning for training. As the dataset is fairly very small I augmented the dataset. Furthermore, as the number of samples is fairly low, the accuracy is very low. To build a more powerful classification model, more data is needed.


Requirements
- Tensorflow
- OpenCV
- Numpy

![A sample of COVID-19 X Ray Image ](covid_case.png)
Format: ![A sample of COVID 19 X Ray Image ]

![A sample of Normal X Ray Image ](normal_case.jpeg)
Format: ![A sample of Normal X Ray Image ]

