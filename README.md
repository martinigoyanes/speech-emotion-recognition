# Dimensional and Discrete Emotion Recognition using Multi-task Learning from Acoustic and Linguistic features extracted from Speech

## Abstract
The majority of research in speech emotion recognition focuses on the classification
of discrete emotions either from acoustic features or text features. This thesis demonstrates that the dimensional representation of emotions is also very valuable and it shows
its advantages over categorical emotions. The thesis proposes two different systems which
both use bimodal features (text and acoustics) in order to recognize discrete and dimensional emotions. A sequential system that first performs dimensional regression and then
classification and a parallel system that performs classification and regression at the same
time.

The thesis develops a multi-task regression model that serves as the core for both systems. Using the Concordance Correlation Coefficient (CCC) for evaluation it is discovered that the thesis developed architecture for dimensional regression outperforms across
all dimensions (valence, arousal, dominance) the regression model introduced in previous
research at the Cambridge institution. In addition, the thesis proves that the sequential
system outperforms the parallel system in the recognition of both discrete (classification
accuracy) and dimensional emotions (CCC). This finding verifies the validity of the theoretical model of psychology about dimensional emotions and its ability to represent a
discrete emotion in a three dimensional space without losing any information. Furthermore, it is demonstrated how transfer learning can be used in this specific task to improve
the results of the system.


## Models 
![](img/mtl-regression.png)  |  ![](img/mtl-class.png)
