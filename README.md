# Motor Imagery Classification
 Motor Imagery Classification for Stroke Rehabilitation


## 1 Introduction
The motor functions were first localized in the motor cortex by Fritsch and Hitzig in the 19th century and subsequently, the experiment replicated by Ferrier provided the same evidence. However, the computational and functional role of the motor cortex has remained a mystery since then. To understand the computational role of the motor cortex, we have implemented an electroencephalography (EEG) based non-invasive method for motor imagery classification using computational models. We will be presenting the mathematical model of each computational algorithm along with the mathematics involved in the preprocessing of data. The model implemented herewith can be used for the rehabilitation of stroke patients, operate BCI-controlled drones, or in virtual/augmented reality applications.       
## 2 Dataset
The dataset used is “A large electroencephalographic motor imagery dataset for electroencephalographic brain-computer interfaces”. We are using the data for one of the paradigms which is the CLA (CLAssic) paradigm. This paradigm includes the data for the BCI interaction model based on three imageries of left-hand motor imagery, right-hand motor imagery, and one passive mental imagery in which the participant is not doing any motor imagery and remains in a neutral state.
The data is acquired using a standard 10/20 EEG cap with a total of 19 bridge electrodes and 2 ground electrodes. EEG signals are sampled at the rate of 200Hz.
### 2.1 Experimental Setup
The participant is presented with the fixation dot at the center of the screen. The stimulus is presented on the screen. For this particular paradigm, there are three stimuli which are images/icons representing left hand, right, and circle for left hand MI, right hand MI and passive state respectively (Fig 1). In each trial, the stimulus is presented for exactly 1s. The left hand MI is implemented by imagining closing and opening of the left-hand fist once. Similarly, right-hand MI is implemented by imagining closing and opening of the right-hand fist once. And for the passive state, the participant does not engage in any motor imagery (remains passive and does not imagine any motor movement voluntarily) until the next trial. After each trial, there is a random off-time which is between 15 to 2.5s.
There are a total of 300 trials per interaction segment (class) and there are a total of three interaction classes, so a total of 900 trials per session. We are currently using the data of three sessions so a total of 2700 trials (for each class there are a total of 900 trials).

## 3 Methods
### 3.1 Preprocessing
#### 3.1.1 Signal Segmentation
The given data in the dataset is a continuous recording of the EEG data in each session. For further processing, we need to first extract the data corresponding to each of the trials. So to do that we need to divide this continuous data in the frames and take out the frame corresponding to each trail. In the given dataset, each point in the signal is marked with corresponding markers (markers for each class and non trial data.) So on the basis of markers, we took out the trail segment for further processing. Each segment is of 170 EEG data points (as the sampling frequency used here is 200Hz).
#### 3.1.2 Signal Denoising
Before further processing, we first need to remove unwanted noise from the signal. After experimenting with the data (according to (Kaya et al., 2018)), the data in the bandwidth of 0 to 5Hz is only relevant for this paradigm. So to filter out the noise from the data we use a low pass Butterworth filter with the cut-off  frequency 5Hz.
### 3.2 Feature Extraction
After removing the noise, we then used this denoised signal for feature extraction. Before extracting the features, we converted the input signal which is in the time domain to the frequency domain. To convert the time domain signal to the frequency domain we used Fourier transform, to be specific we used fast Fourier transform. After converting each  input segment (of size n=170) to the frequency domain we get the segments of size n/2 +1 which are equal to 86 and each data point is an imaginary number. 
### 3.3 Feature Selection
After converting to the frequency domain, as mentioned previously we took only the points in the bandwidth of 0 to 5 Hz, which ultimately gave us a total of 5 features (each representing one frequency bin in Fourier transform) for each segment. As mentioned previously each data point after Fourier transform gives imaginary numbers, so for further proceeding (i.e., for classification), we convert each feature into two features (real part and imaginary part). So for each segment, we eventually get 9 features (imaginary part is zero for 0 frequency bin).
For each trial, as we are taking the data recording from 21 electrodes, the total number of features for each trial will become 189 (21x9). Now we get features from the raw data, we are in a position to classify trials in their respective classes.
### 3.4 Classification
After extraction of the features from raw data, we apply multiple classification algorithms to classify each trail and compare those algorithms against each other. The algorithms used for Classification are as follows:
- Support Vector Machine (SVM)
- Decision Tree   
- Random Forest
- Ada Boosting
- Bagging
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- MLP (Multi-Layer Perceptron)
- QDA (Quadratic Discriminant Analysis)
- KNN (K Nearest Neighbor)

Models like Random forest, Bagging and MLP are giving the highest accuracy over test data and Naive Bayes-based models like Gaussian and Bernoulli Naive Bayes models are giving the lowest accuracy.
### 3.5 Validation
For validation, we validated all the classification models, and found their accuracies on test data.

| Model | Train Accuracy | Best Test Accuracy |
| --- | --- | --- |
| SVM | 99.44 | 99.44 |
| Decision Tree | 98.05 | 95 | 
| Random Forest | 100 | 100 |
| Ada Boosting | 94.58 | 96.11 |
| Bagging | 99.86 | 97.22 |
| GaussianNB | 73.88 | 65 |
| BernoulliNB | 75 | 72.77 |
| MLP | 100 | 99.44 |
| QDA | 99.72 | 85 |
| KNN | 98.47 | 98.33 |


## 4 References
Kaya, M., Binli, M. K., Ozbay, E., Yanar, H., & Mishchenko, Y. (2018). A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces. Scientific Data.

Kaya, M., Binli, M. K., Ozbay, E., Yanar, H., & Mishchenko, Y. (2018). A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces. figshare. https://doi.org/10.6084/m9.figshare.c.3917698.v1

Mishchenko, Y., Kaya, M., Ozbay, E., & Yanar, H. (2019). Developing a Three- to Six-State EEG-Based Brain-Computer Interface for a Virtual Robotic Manipulator Control. IEEE Trans Biomed Eng., 66(4), 977-987. 10.1109/TBME.2018.2865941

Si, Y. (2020). Machine learning applications for electroencephalography signals in epilepsy: a quick review. Acta Epileptologica, 2, 5. https://doi.org/10.1186/s42494-020-00014-0
