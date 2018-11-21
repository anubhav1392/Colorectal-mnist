Overview
The dataset serves as a much more interesting MNIST or CIFAR10 problem for biologists by focusing on histology tiles from patients with colorectal cancer. In particular, 
the data has 8 different classes of tissue (but Cancer/Not Cancer can also be an interesting problem).

About This Code:
This is Version:1 of the classifier model.
In this model i have used InceptionV3 network for feature generation.
As it can be seen in the Metrics Images that model was able to achieve accuracy of 98% but then it starts to overfit a little bit after 12 epochs.
To overcome this issue, i am working on version 2 of this model in which i will use some techniques to reduce this overfitting along with that 
few questions will be answered which were asked on Kaggle:
Which classes are most frequently confused?
What features can be used (like texture) to improve classification?
How can these models be applied to the much larger 5000x5000 models? How can this be done efficiently?
