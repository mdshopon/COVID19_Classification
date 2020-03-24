## COVID19_Classification
This repository contains data for COVID19 and a Transfer Learning Based model for classification

There are two types of classification that were done.

* A four class classification among X-Ray Images. The classes were Nomral X Rays, COVID 19 X Rays, Viral Pneumonia, Bacterial Pneumonia etc. Run <b>four_class.py</b> for this.  
* A two class classification on COVID 19 X rays and Normal X Rays. Run <b>two_class.py</b> for this.

The COVID-19 X-ray image dataset was curated by Dr. Joseph Cohen, a postdoctoral fellow at the University of Montreal. If you want to contribute in his work, please visit his [Github Repository](https://github.com/ieee8023/covid-chestxray-dataset) . The other types of data samples were collected from kaggle and some other sources. 

I used RESNET-50 based transfer learning for training. As the dataset is fairly very small I augmented the dataset. Furthermore, as the number of samples is fairly low, the accuracy is very low. To build a more powerful classification model, more data is needed.

Currently the dataset is divide evenly. 

* Nummber of COVID Samples - 111 ( Train:80, Test: 31) 

To balance the dataset, other class samples were taken according to the number of COVID Samples. 


Requirements
- Tensorflow
- OpenCV
- Numpy

Below there are two samples of X Ray Images 



A sample of COVID-19 X Ray Image           |  A sample of Normal X Ray Image 
:-------------------------:|:-------------------------:
<img src="covid_case.png" width="300" height="300"> |  <img src="normal_case.jpeg" width="300" height="300">
