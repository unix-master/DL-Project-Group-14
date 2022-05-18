# Deep Learning Course Project
This is a course project in the Deep Learning course taken at BITS Goa, supervised by [Dr. Tirtharaj Dash](https://github.com/tirtharajdash).
<br><br>
This repository includes all the relevant files about our project work related to the ICLR 2020 paper
titled [On the relationship between Self-attention and Convolutional Layers](https://arxiv.org/pdf/1911.03584.pdf). The authors have shown
concrete theoretical and empirical evidence of commonality between localized convolution and
self-attention for images. We can argue whether similar results can be obtained for 1D convolution and
attention in the case of sequential data such as audio, text, and time series. In this work, we have made an
effort to extend their idea for multivariate time series data and used standard self-attention Transformer
networks and Deep Convolution Nets to establish this comparison.
<br>
### About the dataset
The Tufts fNIRS [Functional near-infrared spectroscopy](https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html) to Mental Workload dataset contains
brain activity recordings and other data from adult humans performing controlled cognitive workload tasks.
The data was collected for developing completely non-intrusive, implicit brain-computer interfaces that can
accurately detect the current intensity of a person’s mental workload and respond by adjusting an interface
accordingly.

Our task consisted of using multivariate time series data to compare the output of
transformers and convolutional nets easily. The proposed dataset has been effectively used in a few other
unsupervised time series representation learning tasks in brain-computer interfacing. Our primary criteria
were to have a credible and balanced dataset for classification, and this one seemed to check those boxes.
However, we faced a few challenges in using this dataset concerning the main objective of our research
paper.

### Training and Testing Protocol Used
We have adopted a subject-specific modeling scheme wherein we pick one
subject (out of the 68 in the dataset) and train on a subset of that subject’s time-series data. Then, we test the
accuracy of our model on the test set of the same subject. Essentially, we train a separate model for each
subject. Other protocols that could have been used: 
<br>
(1). generic training/testing across all subject
<br>
(2). Training on one subject and testing how it performs on other subjects’ data.

### Contributions
This research paper is an empirical analysis of the relationship between localized convolution and
self-attention based on images. We have looked to extend this idea to sequential data, specifically for
multivariate time-series data. The original paper's authors did mention a possible extension of their
results to 1D convolutions, but they didn’t prove it empirically. We embarked upon the task of doing a
similar analysis but for sequential data involving 1D convolution and 1D self-attention.
