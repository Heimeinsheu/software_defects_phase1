# software_defects_phase1
Dissertation 
CONTENT

0.  	ABSTRACT
I.   	INTRODUCTION
II.  	RELATED WORK 
III. 	EXPERIMENTAL METHODOLOGY
	A.	Datasets and Data Pre-Processing
	B.	Learning Algorithms
D.	 Accuracy Tables
E.	 Neural Networks
F.	Experimental Results
D.	 Accuracy Tables:  
E.	 Neural Networks:
F. 	Experimental Results
IV.	CONCLUSION
V. 	FUTURE WORK
VI.	KEYWORDS	
VII.	REFERENCE
 

 


 

Abstract—Software Engineering is a comprehensive domain since it requires a tight communication between system stake- holders and delivering the system to be developed within a determinate time and a limited budget. Delivering the customer requirements include procuring high performance by minimiz- ing the system. Thanks to effective prediction of system defects on the front line of the project life cycle, the project’s resources and the effort or the software developers can be allocated more efﬁciently for system development and quality assurance activities. The main aim of this paper is to evaluate the capability of machine learning algorithms in software defect prediction and ﬁnd the best category while comparing five machine learning algorithms within the context of five NASA datasets obtained from public PROMISE repository and five more datasets from other sources . 

Keywords—Software quality metrics, Software defect prediction, Software fault prediction, Machine learning algorithms, Neural Networks
I.	INTRODUCTION
Developing a software system is an arduous process which contains planning, analysis, design, implementation, testing, integration and maintenance. A software engineer is expected to develop a software system on time and within limited the budget which are determined during the planning phase. During the development process, there can be some defects such as improper design, poor functional logic, improper data handling, wrong coding, etc. and these defects may cause errors which lead to rework, increases in development and maintenance costs decrease in customer satisfaction. A defect management approach should be applied in order to improve software quality by tracking of these defects. In this approach, defects are categorized depending on the severity and corrective and preventive actions are taken as per the severity deﬁned. Studies have shown that ’defect prevention’ strategies on behalf of ’defect detection’ strategies are used in current methods. Using defect prevention strategies to reduce defects generating during the software development the process is a costly job. It requires more effort and leads to increases in project costs. Accordingly, detecting defects in the software on the front line of the project life cycle is crucial. The implementation of machine learning algorithms which is the binary prediction model enables identify defect- prone modules in the software system before a failure occurs during development process. In this research, our aim is to evaluate the software defect prediction performance of seven machine learning algorithms by utilizing quality metrics; accuracy, precision, recall, F-measure associated with defects as an independent variable and ﬁnd the best category while comparing software defect prediction performance of these
 
machine learning algorithms within the context of four NASA datasets obtained from public PROMISE repository. The selected machine learning algorithms for comparison are used for supervised learning to solve classiﬁcation problems. They are three tree-structured classiﬁer techniques: (i) Bagging, (ii) Random Forests (RF) and (iii) Decision Tree; One Neural networks techniques: (i) Multilayer Perceptron (MLP); and one discriminative classiﬁer Support Vector Machine (SVM). The remainder of the paper is organized as follows: Section 2 brieﬂy describes the related work, while Section 3 describes the experimental methodology in detail. Section 4 contains the conclusion of the experimental study and underlined some possible future research directions.

II.	RELATED WORK
There are a great variety of studies which have developed and applied statistical and machine learning based models for defect prediction in software systems. Basili et al. (1996) [1] have used logistic regression in order to examine what the effect of the suite of object-oriented design metrics is on the prediction of fault-prone classes. Khoshgoftaar et al. (1997)
[7] have used the neural network in order to classify the mod- ules of large telecommunication systems as fault-prone or not and compared it with a non-parametric discriminant model. The results of their study have shown that compared to the non-parametric discriminant model, the predictive accuracy of the neural network model had a better result. Then in 2002 [6], they made a case study by using regression trees to classify fault-prone modules of enormous telecommunication systems. Fenton et al. (2002) [4] have used Bayesian Belief Network in order to identify software defects. However, this machine learning algorithm has lots of limitations which have been recognized by Weaver(2003) [14] and Ma et al. (2007) [9]. Guo et al. (2004) [5] have applied Random Forest algorithm on software defect dataset introduced by NASA to predict fault-prone modules of software systems and compared their model with some statistical and machine learning models. The result of this comparison has shown that compared to other methods, the random forest algorithm has given better predictive accuracy. Ceylan et al. (2006) [2] have proposed a model which uses three machine learning algorithms that are Decision Tree, Multilayer Perceptron and Radial Basis Functions in order to identify the impact of this model to predict defects on different software metric datasets obtained from the real*life projects of three big-size software companies in Turkey. The results have shown that
 

 
all of the machine learning algorithms had similar results which have enabled to predict potentially defective software and take actions to correct them. Elish et al. (2008) [3] have investigated the impact of Support Vector Machines on four NASA datasets to predict defect-proneness of software systems and compared the prediction performance of SVM against eight statistical and machine learning models. The results have indicated that the prediction performance of SVM has been much better than others. Kim et al. (2011) [8] have investigated the impact of the noise on defect prediction to cope with the noise in defect data by using a noise detection and elimination algorithm. The results of the study have presented that noisy instances could be predicted with reasonable accuracy and applying elimination has improved the defect prediction accuracy. Wang at all. (2013) [13] have investigated re-sampling techniques, ensemble algorithms and threshold moving as class imbalance learning methods for software defect prediction. They have used different methods and among them, AdaBoost.NC had better defect prediction performance. They have also improved the effectiveness and efﬁciency of AdaBoost.NC by using a dynamic version of it. Ren at al. (2014) [11] have proposed a model to solve the class imbalance problem which causes a reduction in the performance of defect prediction. The Gaussian function has been used as kernel function for both the Asymmetric Kernel Partial Least Squares Classiﬁer (AKPLSC) and Asymmetric Kernel Principal Component Analysis Classiﬁer (AKPCAC) and NASA and SOFTLAB datasets have been used for experiments. The results have shown that the AKPLSC had better impact on retrieving the loss caused by class imbalance and the AKPCAC had better performance to predict defects on imbalanced datasets. There is also a systematic review study conducted by Malhotra to review the machine learning algorithms for software fault prediction.
III.	EXPERIMENTAL    METHODOLOGY
A.	Datasets and Data Pre-Processing
The datasets which are available from the public PROMISE repository and used for this task are detailed in Figure I and the other datasets are detailed in Figure II. These datasets have different number of instances. The dataset with the most data in terms of the number of instances is JM1 with 10879 instances. Data sets of different sizes have been selected to demonstrate the effect of data size on accuracy. In Table I, each dataset explained with language, number of attributes, number of instances, percentage of defective modules and description. The number of attributes is 22 for KC1, KC2, CM1, PC1 and JM1 datasets and 30 for AR1, AR3, AR4, AR5 and AR6 datasets. Further we check for Null values
kc1_df.isnull().sum()
Checking type of each attribute in dataset. Converting
attribute of following dataset into required datatype. While converting attribute having object type into numeric type. Encounter many rows contain “?” value in respective attribute having object datatype. Instead of null “?” was used. Scaling datasets.
Attribute information is shown in Figure I And Figure II.
   








 Fig I. ATTRIBUTE DEFINITION FOR NASA DATASET
	 
  
   Fig II. ATTRIBUTE DEFINITION FOR SOFTLAB DATASET
	 

We noted a huge class imbalance issue with the available datasets (faulty, non-faulty) as revealed in the figures below(Fig III to Fig X) which can cause high bias and lead to wrong prediction. We have used several methods to counter class imbalance.

K-fold Cross-Validation (CV) model is employed for each learning algorithm to model validation. The k value is deter- mined as 10 in this experiment. Since the number of samples
in the used datasets are equal to 10, the data is divided into 10 folds. That means k-1 objects in the dataset are used as training samples and one object is used as test sample in the each iteration. That is, every data fold is used as a validation set exactly once and falls into a training set k-1 times. Then the average error across all k trials which is equal to the number of samples in the dataset is computed. 

SMOTE (Synthetic Minority Oversampling Technique) is a data augmentation technique that helps to address the issue of imbalanced classes in machine learning datasets. It works by creating synthetic data points that are similar to the minority class data points, in order to balance the dataset and improve the performance of machine learning models.

Stratified Sampling is a sampling method that reduces the sampling error in cases where the population can be partitioned into subgroups. We perform Stratified Sampling 



 

by dividing the population into homogeneous subgroups, called strata, and then applying Simple Random Sampling within each subgroup.

Shuffle Split Unlike K-Fold, Shuffle Split leaves out a percentage of the data, not to be used in the train or validation sets. To do so we must decide what the train and test sizes are, as well as the number of splits.

Fig  III: Class imbalance JM1 
 

Fig IV: Class imbalance KC1
 

Fig V: Class imbalance KC2
 
Fig VI: Class imbalance CM1
 
Fig VIII: Class imbalance PC1
 



Fig IX: Class imbalance AR1
 
Fig X: Class imbalance AR3
 

B.	Learning Algorithms
In this experiment, we have used for defect prediction in software systems. They categorized the machine learning algorithms based on distinct learners such as Ensemble Learners, Neural Networks and SVM. According to these categories, we selected five different machine learning algorithms to estimate software defect. Each algorithm is detailed below in Table I.

 




 
     						TABLE I. DATASET PROPERTIES
		Project	Language	# of Attributes	# of instances	% of Defective Modules	Description
Procedural
(NASA)		CM1	C	22	498	9.7	CM1 is a NASA spacecraft instrument written in ”C”.
		PC1
JM1
	C
C	22
22	1109
10879	6.9
19.35	Data from C functions. It is a ﬂight software developed for earth orbiting satellite.

 Procedural
(Softlab)		AR1
AR3
AR4
AR5
AR6	C
C
C
C
C	30
30
30
30
30	120
62
106
35
100		{ Softlab 
Data from a Turkish white-goods manufacturer .
Embedded software implemented in C. Function/method level static code attributes are collected using the Preset Metrics Extraction and Analysis Tool.}
Object Oriented
(NASA)		KC1	C++	22	2109	15.4	KC1 is a system written by using C++ programming language. It implements storage management in order
	
KC2	
Java	                                                                                                                                      to receive and process ground data.	

22	522	6.3	Data obtained from C++ functions. KC2 is a system
developed for science data processing. It was developed by different developers than KC1 project as an extension of it. In this implementation, only some third-party software libraries of KC1 were used, the remainder of the software was developed differently.

 
 
   1)Ensemble Learners:
•	Bagging: This algorithm which is introduced by Leo Breiman and also called Bootstrap Aggregation is one of the ensemble methods. In this approach, N sub-samples of data from the training sample are created and the predictive model is trained by using these subset data. Sub-samples are chosen randomly with replacement. As a result, the ﬁnal model is an ensemble of different models.
•	Random Forest: Random Forest algorithms which also called random decision forest is an ensemble tree-based learning algorithm. It makes a prediction over individual trees and selects the best vote of all predicted classes over trees to reduce overﬁtting and improve generaliza tion accuracy. It is also the most ﬂexible and easy to use for both classiﬁcation and regression.
•	Decision Tree: Decision Tree algorithm is a supervised learning technique that can be used for both classification and regression problems. It is a tree-structured classifier where internal nodes represent the features of a dataset, branches represent the decision rules, and each leaf node represents the outcome.
  2) Neural Networks:
•	Simple Perceptron: The Simple Perceptron is a basic type of artificial neural network that consists of a single layer of artificial neurons, also known as perceptron .The Simple Perceptron is a binary classifier that learns from labeled training data to make predictions. 
•	Multilayer Perceptron:   Multilayer Perceptron which is one of the types of Neural Networks comprises of one input layer, one output layer and at least one or more hidden layers. This algorithm transfers the data from the input layer to the output layer, which is called     feed forward.
•	Multilayer Neural Network + Permutation: Similiar to MLNN, layers and neurons in each layer are arranged in permutation manner.

3) Support Vector Machines:
 Support vector machine (SVM) is a supervised machine learning method capable of both classification and regression. It is one of the most effective and simple methods used in classification. For classification, it is possible to separate two groups by drawing decision boundaries between two classes of data points in a hyperplane. The main objective of this algorithm is to find optimal hyperplane.
C.	C) Evaluation Metrics
To evaluate learning algorithms which are stated above, commonly used evaluation metrics are used such as accuracy, precision, recall, F-measure. The performance of the model of each algorithm is evaluated by using the confusion matrix which is called as an error matrix and is a summary of prediction results on a classification problem. Evaluation of model is the most important for classification problem where the output can be of two or more types of classes and the confusion matrix is one of the most commonly used and easiest metrics for determining the accuracy of the model. It has True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN) values.
• Positive (P) : Observation is positive (for example: is an
defective).
• Negative (N) : Observation is not positive (for example:
is not an defective).
• True Positive (TP) : The model has estimated true and
the test data is true.
• False Negative (FN) : The model has estimated false and
the test data is true.
• True Negative (TN) : The model has estimated false and
the test data is false.
• False Positive (FP) : The model has estimated true and
the test data is false.
 

1) Accuracy: Accuracy which is called classification rate
is given by the following relation:
Accuracy =(TP + TN)/(TP + TN + FP + FN)	(1)
2) Recall: To get the value of Recall, correctly predicted
positive observations is divided by the all observations in
actual class and it can be defined as below:
Recall =TP/(TP + FN)			(3)
3) Precision: Precision is the ratio of the total number
of correctly classified positive examples to the number of
predicted positive examples. As shown in Equation 4, As
decreases the value of FP, precision increases and it indicates
an example labeled as positive is indeed positive.
Precision = TP/(TP + FP)			(4)



4) F-measure: Unlike recall and precision, this metric
takes into account both false positives(FP) and false negatives(
FN). F-measure is the weighted harmonic mean of the
precision and recall of the test. The equation of this metric
is shown in Equation 5.
Precision = (2 ∗ Recall ∗ Precision)/(Recall + Precision)       (5)


D) Accuracy Tables:  The below tables shows the accuracy given by machine learning algorithms datasets with respective datasets.
 

 
Table II: Machine Learning Algorithms
	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6
Linear Regression	0.8285714	0.8483412	0.93	0.9234234	0.8143382	0.875	0.8461538	0.9090909	1	0.85
Suppport Vector Machine	0.8952381	0.8459716	0.93	0.9324324	0.8157169	0.875	0.9230769	0.8636364	0.7142857	0.85
Decision Tree	0.8095238	0.8388626	0.88	0.9369369	0.7601103	0.875	0.9230769	0.8181818	0.7142857	0.75
Random Forest	0.7904762	0.8554502	0.92	0.9324324	0.7601103	0.875	0.9230769	0.8181818	0.7142857	0.8
Bagging	0.8380952	0.8175355	0.89	0.9234234	0.7555147	0.8333333	0.9230769	0.7727273	0.8571429	0.7
										
Maximum	0.895238	0.85545	0.93	0.936937	0.815717	0.875	0.923077	0.909091	1	0.85
Conclusion of Table II:
Support Vector Machine (SVM) algorithm achieved the highest performance in most datasets, followed by Linear Regression. Decision Tree and Random Forest had comparable results, while Bagging showed slightly lower performance overall.
Table III: K-Cross Validation with Machine Learning Algorithms

K- values	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6
lr2	0.82343796	0.85199241	0.87928326	0.92689531	0.7777401	0.9	0.80645161	0.77358491	0.82679739	0.84
lr3	0.82537151	0.84250782	0.88938785	0.9323128	0.77249438	0.89166667	0.79126984	0.78306878	0.85606061	0.83006536
lr4	0.823532	0.84392789	0.88733871	0.93050542	0.78637941	0.91666667	0.81041667	0.80235043	0.82986111	0.83
lr5	0.82738095	0.85056793	0.88537374	0.93054502	0.76468327	0.9	0.87051282	0.8021645	0.88571429	0.85
lr6	0.82339838	0.84346267	0.88938682	0.93143361	0.78205513	0.90833333	0.80909091	0.82189542	0.83333333	0.82843137
lr7	0.82931789	0.84962141	0.89336016	0.93052191	0.77810253	0.89169001	0.84126984	0.81130952	0.88571429	0.85034014
lr8	0.82934149	0.84630466	0.88524066	0.92512512	0.771945	0.9	0.88839286	0.8282967	0.85	0.84054487
lr9	0.81955367	0.84915237	0.88329726	0.93322552	0.77102372	0.89133089	0.87301587	0.81986532	0.86111111	0.83922559
lr10	0.83120464	0.84582261	0.88546939	0.93140868	0.80126884	0.9	0.87380952	0.84	0.86666667	0.86
svm2	0.82533893	0.84535104	0.90142829	0.92870036	0.80853047	0.91666667	0.90322581	0.83018868	0.74509804	0.84
svm3	0.83688791	0.85009774	0.90146039	0.93050856	0.8082543	0.91666667	0.90396825	0.83015873	0.74494949	0.84997029
svm4	0.82729008	0.85104364	0.90145161	0.93050542	0.80825485	0.91666667	0.88854167	0.85861823	0.77430556	0.87
svm5	0.83302198	0.8529376	0.90145455	0.93054502	0.80807057	0.91666667	0.90384615	0.83896104	0.8	0.87
svm6	0.83110576	0.85058113	0.90143501	0.93053271	0.80834708	0.91666667	0.9030303	0.84858388	0.77777778	0.8682598
svm7	0.82924067	0.85151513	0.90140845	0.9305276	0.80834771	0.91690009	0.9047619	0.84880952	0.71428571	0.86938776
svm8	0.8291958	0.85199548	0.90136969	0.93054035	0.80816283	0.91666667	0.90625	0.84752747	0.69375	0.86698718
svm9	0.82913222	0.85057687	0.90147908	0.9305082	0.8081627	0.91697192	0.9047619	0.84848485	0.72222222	0.86952862
svm10	0.83494194	0.8097723	0.90155102	0.93052416	0.80816247	0.91666667	0.9047619	0.84909091	0.73333333	0.87
dta2	0.80422193	0.82969744	0.83898497	0.90794224	0.74170477	0.86666667	0.79032258	0.82075472	0.7745098	
dta3	0.78703519	0.81593928	0.83699647	0.90793232	0.75052875	0.9	0.82222222	0.8010582	0.8030303	0.81966726
dta4	0.79095713	0.82733055	0.831	0.89711191	0.75273447	0.875	0.82291667	0.80163818	0.82986111	0.79
dta5	0.80813187	0.81927178	0.8429899	0.89984917	0.75300972	0.85833333	0.83846154	0.76363636	0.82857143	0.78
dta6	0.81564644	0.82162894	0.84712019	0.90796612	0.75080491	0.85	0.8530303	0.79357298	0.77777778	0.75
dta7	0.79660232	0.82780274	0.84909457	0.89710214	0.75218474	0.85901027	0.83730159	0.74404762	0.77142857	0.80952381
dta8	0.79090909	0.83397789	0.85093446	0.90975263	0.7571472	0.85833333	0.85267857	0.76373626	0.83125	0.77804487
dta9	0.79276736	0.82683593	0.8530303	0.90072704	0.75631835	0.86691087	0.83597884	0.77441077	0.72222222	0.78956229
dta10	0.78320029	0.86966825	0.84110204	0.90339885	0.74804533	0.85833333	0.83809524	0.77545455	0.78333333	0.75
rf2	0.83687003	0.85483871	0.87928326	0.93411552	0.80797895	0.91666667	0.91935484	0.83018868	0.80228758	0.82
rf3	0.83688791	0.85579034	0.88938785	0.93592129	0.80926571	0.90833333	0.90396825	0.81984127	0.80050505	0.84997029
rf4	0.83122431	0.84440228	0.89337097	0.93231047	0.80954152	0.9	0.90416667	0.83938746	0.86111111	0.85
rf5	0.82545788	0.86195923	0.88735354	0.93142554	0.80954116	0.90833333	0.90384615	0.81038961	0.88571429	0.85
rf6	0.83680834	0.85674858	0.89536194	0.9314434	0.81027693	0.91666667	0.88787879	0.8583878	0.83333333	0.81801471
rf7	0.82545689	0.85294053	0.89939638	0.93592412	0.81092172	0.91736695	0.87301587	0.8202381	0.8	0.86938776
rf8	0.84460956	0.8543683	0.89330517	0.92692368	0.81156323	0.91666667	0.88839286	0.8282967	0.8	0.84775641
rf9	0.82909861	0.86006951	0.88333333	0.93502491	0.80788737	0.8998779	0.9047619	0.81902357	0.80555556	0.8493266
rf10	0.84259797	0.85389077	0.88342857	0.92961507	0.80760965	0.90833333	0.92142857	0.84909091	0.78333333	0.83
b2	0.84645594	0.85199241	0.88934771	0.93411552	0.80614029	0.9	0.87096774	0.82075472	0.7745098	0.82
b3	0.82534937	0.85199504	0.88739199	0.93321126	0.80862244	0.90833333	0.91984127	0.82936508	0.77272727	0.84997029
b4	0.84270405	0.85246679	0.88332258	0.92599278	0.80926527	0.91666667	0.90520833	0.83012821	0.82986111	0.85
b5	0.84849817	0.84819939	0.88137374	0.92963597	0.81597608	0.89166667	0.90384615	0.81038961	0.82857143	0.86
b6	0.84257774	0.85532407	0.89940249	0.93774481	0.81174839	0.91666667	0.91969697	0.81100218	0.72222222	0.83823529
b7	0.82540541	0.8482023	0.88531187	0.93322301	0.81266771	0.91690009	0.92063492	0.81071429	0.82857143	0.85102041
b8	0.83114802	0.85911215	0.90335381	0.93323168	0.80990909	0.91666667	0.91964286	0.83791209	0.83125	0.86858974
b9	0.82725012	0.92063492	0.90151515	0.93141155	0.80990948	0.9004884	0.9047619	0.83754209	0.83333333	0.85942761
b10	0.83693759	0.84914241	0.89146939	0.93230139	0.81000105	0.91666667	0.9047619	0.81	0.84166667	0.85

Table IV:Summary of  K-Cross Validation with Machine Learning Algorithms
	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6		Percentage
Linear Regression	0.8312	0.852	0.8934	0.93323	0.80127	0.9167	0.8884	0.84	0.8857	0.86		10%
Suppport Vector Machine	0.83689	0.8529	0.9016	0.93055	0.80853	0.917	0.9063	0.85862	0.8	0.87		20%
Decision Tree	0.81565	0.8697	0.853	0.90975	0.75715	0.9	0.7571	0.82075	0.8313	0.81967		0%
Random Forest	0.84461	0.862	0.8994	0.93592	0.81156	0.9174	0.9214	0.85839	0.8857	0.86939		20%
Bagging	0.8485	0.8485	0.8485	0.8485	0.8485	0.9169	0.9206	0.83791	0.8417	0.86859		50%
												
Maximum	0.8485	0.8697	0.9016	0.93592	0.8485	0.9174	0.9214	0.85862	0.8857	0.87		

Conclusion of Table IV:
Bagging achieved the highest performance in most metrics, followed by Random Forest. Support Vector Machine and Linear Regression showed comparable results, while Decision Tree had slightly lower performance overall.

Table V: SMOTE with K-Cross Validation with Machine Learning Algorithms

	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6
lr2	0.82343796	0.84914611	0.87928326	0.92689531	0.7777401	0.9	0.80645161	0.77358491	0.82679739	0.84
lr3	0.82728722	0.8463065	0.88938785	0.9323128	0.77249438	0.89166667	0.79126984	0.78306878	0.85606061	0.83006536
lr4	0.82354668	0.84392789	0.88733871	0.93050542	0.78637941	0.91666667	0.81041667	0.80235043	0.82986111	0.83
lr5	0.82738095	0.85056793	0.88537374	0.93054502	0.76468327	0.9	0.87051282	0.8021645	0.88571429	0.85
lr6	0.82148267	0.84393751	0.88938682	0.93143361	0.78205513	0.90833333	0.80909091	0.82189542	0.83333333	0.82843137
lr7	0.83312741	0.84961984	0.89336016	0.93052191	0.77810253	0.89169001	0.84126984	0.81130952	0.88571429	0.85034014
lr8	0.83126457	0.84677454	0.88524066	0.92512512	0.771945	0.9	0.88839286	0.8282967	0.85	0.84054487
lr9	0.81955367	0.8467883	0.88329726	0.93322552	0.77102372	0.89133089	0.87301587	0.81986532	0.86111111	0.83922559
lr10	0.83120464	0.84819228	0.88546939	0.93140868	0.80126884	0.9	0.87380952	0.84	0.86666667	0.86
svm2	0.82533893	0.84535104	0.90142829	0.92870036	0.80853047	0.91666667	0.90322581	0.83018868	0.74509804	0.84
svm3	0.831105765	0.85009774	0.90146039	0.93050856	0.8082543	0.91666667	0.90396825	0.83015873	0.74494949	0.84997029
svm4	0.82729008	0.85104364	0.90145161	0.93050542	0.80825485	0.91666667	0.88854167	0.85861823	0.77430556	0.87
svm5	0.83302198	0.8529376	0.90145455	0.93054502	0.80807057	0.91666667	0.90384615	0.83896104	0.8	0.87
svm6	0.83110576	0.85058113	0.90143501	0.93053271	0.80834708	0.91666667	0.9030303	0.84858388	0.77777778	0.8682598
svm7	0.82924067	0.85151513	0.90140845	0.9305276	0.80834771	0.91690009	0.9047619	0.84880952	0.71428571	0.86938776
svm8	0.8291958	0.85199548	0.90136969	0.93054035	0.80816283	0.91666667	0.90625	0.84752747	0.69375	0.86698718
svm9	0.82913222	0.85057687	0.90147908	0.9305082	0.8081627	0.91697192	0.9047619	0.84848485	0.72222222	0.86952862
svm10	0.83494194	0.852478	0.90155102	0.93052416	0.80816247	0.91666667	0.9047619	0.84909091	0.73333333	0.87
dta2	0.81000589	0.8097723	0.83898497	0.90794224	0.74170477	0.86666667	0.79032258	0.82075472	0.7745098	0.79
dta3	0.77745665	0.82969744	0.83699647	0.90793232	0.75052875	0.9	0.82222222	0.8010582	0.8030303	0.81966726
dta4	0.79094245	0.82969744	0.831	0.89711191	0.75273447	0.875	0.82291667	0.80163818	0.82986111	0.79
dta5	0.80815018	0.82733055	0.8429899	0.89984917	0.75300972	0.85833333	0.83846154	0.76363636	0.82857143	0.78
dta6	0.80412991	0.81927178	0.84712019	0.90796612	0.75080491	0.85	0.8530303	0.79357298	0.77777778	0.75
dta7	0.79655084	0.82162894	0.84909457	0.89710214	0.75218474	0.85901027	0.83730159	0.74404762	0.77142857	0.80952381
dta8	0.79664918	0.82780274	0.85093446	0.90975263	0.7571472	0.85833333	0.85267857	0.76373626	0.83125	0.77804487
dta9	0.79848088	0.83397789	0.8530303	0.90072704	0.75631835	0.86691087	0.83597884	0.77441077	0.72222222	0.78956229
dta10	0.79470247	0.82683593	0.84110204	0.90339885	0.74804533	0.85833333	0.83809524	0.77545455	0.78333333	0.75
rf2	0.84647067	0.85294118	0.88133178	0.93501805	0.81082857	0.91666667	0.90322581	0.85849057	0.85784314	0.82
rf3	0.83499435	0.847731	0.89340392	0.93682219	0.81110447	0.9	0.90396825	0.81058201	0.82828283	0.8600713
rf4	0.823532	0.85958254	0.89741935	0.93140794	0.80944947	0.91666667	0.88854167	0.83903134	0.80555556	0.85
rf5	0.81197802	0.85293985	0.88337374	0.93233052	0.80742812	0.91666667	0.88717949	0.81082251	0.82857143	0.84
rf6	0.84645371	0.85200563	0.90750808	0.93502742	0.80650856	0.91666667	0.90454545	0.82026144	0.86111111	0.83823529
rf7	0.82342342	0.85056434	0.89336016	0.93503133	0.80981804	0.90849673	0.9047619	0.8202381	0.8	0.87891156
rf8	0.82931235	0.8576971	0.89125704	0.92870269	0.81000174	0.90833333	0.91964286	0.80906593	0.8625	0.84775641
rf9	0.82540163	0.85295508	0.89339827	0.93049363	0.81202272	0.8998779	0.9047619	0.80050505	0.83333333	0.8493266
rf10	0.84071118	0.85911532	0.89755102	0.92961507	0.81386117	0.91666667	0.92142857	0.80181818	0.86666667	0.84
b2	0.83302387	0.85436433	0.88333171	0.93592058	0.80797902	0.91666667	0.90322581	0.83962264	0.74509804	0.83
b3	0.81385511	0.84298333	0.88937568	0.92960277	0.80917368	0.9	0.88650794	0.83888889	0.85606061	0.8600713
b4	0.84458309	0.85436433	0.90345161	0.93140794	0.80852988	0.91666667	0.93645833	0.83938746	0.80555556	0.85
b5	0.83501832	0.85721088	0.89547475	0.93142962	0.81009267	0.91666667	0.92051282	0.81991342	0.8	0.85
b6	0.8387686	0.859116	0.88936233	0.93593811	0.8082548	0.91666667	0.91969697	0.83877996	0.83333333	0.84742647
b7	0.82355212	0.86053898	0.89537223	0.93140901	0.81230035	0.90009337	0.92063492	0.80119048	0.85714286	0.87006803
b8	0.82537879	0.85721461	0.88930492	0.93503675	0.80825468	0.9	0.90401786	0.84752747	0.85625	0.83814103
b9	0.82909861	0.85153462	0.89336219	0.93140426	0.80687575	0.90903541	0.92063492	0.81902357	0.83333333	0.80976431
b10	0.82155298	0.85862334	0.89546939	0.93501229	0.81257534	0.91666667	0.92142857	0.83	0.81666667	0.84


Table VI: Summary of  SMOTE with K-Cross Validation with Machine Learning Algorithms

	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6		Percentage
Linear Regression	0.83313	0.8506	0.8934	0.9332	0.80127	0.9167	0.8884	0.84	0.8857	0.86		10%
Suppport Vector Machine	0.83494	0.8529	0.9016	0.9305	0.80853	0.917	0.9063	0.85862	0.8	0.87		20%
Decision Tree	0.81001	0.834	0.853	0.9098	0.75715	0.9	0.853	0.82075	0.8313	0.81967		0%
Random Forest	0.84647	0.8596	0.9075	0.9368	0.81386	0.9167	0.9214	0.85849	0.8667	0.87891		50%
Bagging	0.84458	0.8605	0.9035	0.9359	0.81258	0.9167	0.9365	0.84753	0.8571	0.87007		20%
												
Maximum	0.84647	0.8605	0.9075	0.9368	0.81386	0.917	0.9365	0.85862	0.8857	0.87891		

Conclusion of Table VI:
Random Forest achieved the highest performance in most metrics, followed by Bagging. Support Vector Machine and Linear Regression showed comparable results, while Decision Tree had slightly lower performance overall.

Table VII: Shuffle Split with K-Cross Validation with Machine Learning Algorithms
	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6
lr2	0.84394904	0.84202212	0.90333333	0.92642643	0.74846814	0.86111111	0.81578947	0.84375	0.81818182	0.9
lr3	0.82802548	0.83307004	0.89111111	0.92692693	0.765625	0.84259259	0.85964912	0.8125	0.84848485	0.76666667
lr4	0.84235669	0.83886256	0.90833333	0.92192192	0.76263787	0.84027778	0.81578947	0.8046875	0.70454545	0.81666667
lr5	0.82802548	0.85339652	0.89066667	0.92912913	0.79203431	0.81111111	0.81052632	0.7875	0.8	0.83333333
lr6	0.83014862	0.85334387	0.87444444	0.93043043	0.78962418	0.85185185	0.90350877	0.80208333	0.81818182	0.85
lr7	0.81619654	0.84269916	0.88571429	0.93393393	0.77376576	0.87698413	0.87969925	0.83035714	0.83116883	0.84285714
lr8	0.81449045	0.8449842	0.88333333	0.92342342	0.79878983	0.87847222	0.90131579	0.80859375	0.875	0.8375
lr9	0.81670205	0.84500614	0.88592593	0.93693694	0.7885689	0.88271605	0.83625731	0.82291667	0.7979798	0.87037037
lr10	0.84522293	0.84802528	0.89133333	0.92972973	0.79286152	0.89722222	0.83684211	0.81875	0.77272727	0.86666667
svm2	0.82802548	0.8507109	0.91333333	0.93393393	0.80713848	0.88888889	0.86842105	0.796875	0.81818182	0.81666667
svm3	0.83864119	0.85202738	0.90666667	0.92992993	0.80831291	0.94444444	0.87719298	0.79166667	0.72727273	0.82222222
svm4	0.81210191	0.86176935	0.92333333	0.92792793	0.80958946	0.90972222	0.89473684	0.8515625	0.79545455	0.81666667
svm5	0.83821656	0.85308057	0.89333333	0.92012012	0.81023284	0.93888889	0.86315789	0.8625	0.83636364	0.88
svm6	0.82696391	0.85255398	0.89111111	0.92592593	0.80744485	0.91666667	0.85964912	0.84375	0.77272727	0.86111111
svm7	0.83348499	0.84608441	0.90190476	0.92706993	0.80435924	0.92063492	0.90225564	0.84375	0.76623377	0.86190476
svm8	0.84235669	0.84695893	0.90416667	0.9298048	0.80932138	0.93055556	0.86184211	0.8671875	0.78409091	0.88333333
svm9	0.8393489	0.85518694	0.90814815	0.93360027	0.81076389	0.91975309	0.88888889	0.80902778	0.78787879	0.84814815
svm10	0.83566879	0.85402844	0.90666667	0.92762763	0.80821078	0.93055556	0.85789474	0.821875	0.74545455	0.82666667
dta2	0.76751592	0.81674566	0.85	0.91741742	0.75076593	0.86111111	0.84210526	0.796875	0.86363636	0.8
dta3	0.80042463	0.83043707	0.83777778	0.9029029	0.74632353	0.89814815	0.77192982	0.83333333	0.78787879	0.83333333
dta4	0.80414013	0.83017378	0.84666667	0.91366366	0.74869792	0.83333333	0.92105263	0.8359375	0.70454545	0.79166667
dta5	0.80382166	0.81895735	0.83066667	0.9033033	0.7504902	0.89444444	0.88421053	0.73125	0.69090909	0.8
dta6	0.77707006	0.82464455	0.84444444	0.9049049	0.75270629	0.89351852	0.85087719	0.765625	0.74242424	0.79444444
dta7	0.78980892	0.82171067	0.83809524	0.91334191	0.75157563	0.82539683	0.93984962	0.76785714	0.83116883	0.74285714
dta8	0.79378981	0.81516588	0.845	0.91591592	0.75214461	0.89930556	0.82236842	0.78125	0.80681818	0.77083333
dta9	0.79688606	0.81323504	0.82518519	0.90824157	0.74176198	0.85493827	0.81871345	0.78819444	0.84848485	0.73333333
dta10	0.79171975	0.82006319	0.84533333	0.91771772	0.7557598	0.88055556	0.85263158	0.8	0.8	0.73666667
rf2	0.78980892	0.85308057	0.88666667	0.92642643	0.80193015	0.90277778	0.86842105	0.875	0.72727273	0.83333333
rf3	0.8089172	0.85150079	0.88444444	0.92592593	0.80637255	0.88888889	0.89473684	0.86458333	0.87878788	0.83333333
rf4	0.82961783	0.85189573	0.89466667	0.9451952	0.80974265	0.91666667	0.94736842	0.8203125	0.77272727	0.88333333
rf5	0.82420382	0.84897314	0.89466667	0.93153153	0.80563725	0.93333333	0.87368421	0.86875	0.89090909	0.82
rf6	0.83333333	0.84702475	0.88888889	0.92692693	0.80907884	0.89351852	0.88596491	0.86979167	0.8030303	0.88333333
rf7	0.82165605	0.84856691	0.89428571	0.93951094	0.8075105	0.90079365	0.93984962	0.85714286	0.79220779	0.81904762
rf8	0.82643312	0.8485387	0.8875	0.93768769	0.80713848	0.89930556	0.88157895	0.84765625	0.79545455	0.80833333
rf9	0.82661005	0.85132526	0.89481481	0.93293293	0.80701934	0.90123457	0.88304094	0.85069444	0.85858586	0.84814815
rf10	0.82675159	0.8521327	0.89466667	0.93453453	0.80882353	0.91666667	0.87894737	0.834375	0.80909091	0.83
b2	0.8343949	0.84123223	0.90333333	0.93093093	0.80392157	0.84722222	0.84210526	0.78125	0.72727273	0.83333333
b3	0.8343949	0.85150079	0.89333333	0.94594595	0.81188725	0.90740741	0.87719298	0.86458333	0.81818182	0.83333333
b4	0.83121019	0.84913112	0.90333333	0.94069069	0.80200674	0.91666667	0.85526316	0.84375	0.93181818	0.85
b5	0.80254777	0.85276461	0.888	0.93753754	0.81004902	0.90555556	0.96842105	0.85625	0.81818182	0.86
b6	0.83545648	0.84570827	0.88888889	0.94044044	0.81081495	0.90277778	0.89473684	0.84375	0.74242424	0.88333333
b7	0.8189263	0.84653577	0.88666667	0.94337194	0.81083683	0.91269841	0.91729323	0.80357143	0.77922078	0.85238095
b8	0.82404459	0.85604265	0.89416667	0.93168168	0.80832567	0.90277778	0.90789474	0.8671875	0.86363636	0.84166667
b9	0.82519462	0.84799017	0.90074074	0.93393393	0.80712146	0.91049383	0.85964912	0.84375	0.73737374	0.83333333
b10	0.82165605	0.84723539	0.88533333	0.93723724	0.80916054	0.88888889	0.91052632	0.825	0.81818182	0.84333333






Table VIII: Summary of Shuffle Split with K-Cross Validation with Machine Learning Algorithms

	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6		Percentage
Linear Regression	0.8452	0.8534	0.9083	0.9369	0.79879	0.89722	0.9035	0.8438	0.875	0.9		20%
Suppport Vector Machine	0.8424	0.8618	0.9233	0.9339	0.81076	0.94444	0.9023	0.8672	0.83636	0.8833		30%
Decision Tree	0.8041	0.8304	0.85	0.9177	0.75576	0.89931	0.9398	0.8359	0.86364	0.8333		0%
Random Forest	0.8333	0.8531	0.8948	0.9452	0.80974	0.93333	0.9474	0.875	0.89091	0.8833		10%
Bagging	0.8355	0.856	0.9033	0.9459	0.81189	0.91667	0.9684	0.8672	0.93182	0.8833		40%
												
Maximum	0.8452	0.8618	0.9233	0.9459	0.81189	0.94444	0.9684	0.875	0.93182	0.9		

Conclusion of Table VIII:
Decision Tree algorithm achieved the lowest performance across all metrics, with a percentage of 0%. Support Vector Machine and Bagging algorithms had the highest performance in various metrics, followed by Linear Regression and Random Forest.
Table IX: Stratified with Machine Learning Algorithms 

	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6
Linear Regression	0.7904762	0.8601896	0.9	0.9279279	0.8092831	0.9583333	0.9230769	0.9090909	0.8571429	0.85
Suppport Vector Machine	0.8095238	0.8554502	0.9	0.9459459	0.8111213	0.9166667	0.9230769	0.8636364	1	0.85
Decision Tree	0.7904762	0.8009479	0.77	0.9189189	0.7522978	0.9166667	0.9230769	0.8636364	0.8571429	0.7
Random Forest	0.8190476	0.8578199	0.9	0.9459459	0.8046875	0.9166667	1	0.8636364	0.7142857	0.85
Bagging	0.7333333	0.8056872	0.8	0.9099099	0.7582721	0.8333333	1	0.9090909	0.8571429	0.8
										
Maximum	0.819048	0.86019	0.9	0.945946	0.811121	0.958333	1	0.909091	1	0.85

Conclusion of Table IX:
Support Vector Machine algorithm achieved the highest performance in most metrics, followed closely by Random Forest. Linear Regression and Decision Tree showed comparable results, while Bagging had slightly lower performance overall.
Table X: Stratified with K-Cross Validation with Machine Learning Algorithms
	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6
lr2	0.79076039	0.85341556	0.88328313	0.9133574	0.76404498	0.81666667	0.88709677	0.79245283	0.82679739	0.75
lr3	0.81367794	0.84346155	0.88531094	0.92599917	0.76671369	0.81666667	0.88730159	0.82063492	0.7979798	0.79976233
lr4	0.82332648	0.84487666	0.88129032	0.92509025	0.75972632	0.84166667	0.83854167	0.79131054	0.79513889	0.79
lr5	0.83087912	0.84441355	0.88537374	0.92509478	0.77020436	0.85833333	0.84230769	0.8017316	0.82857143	0.81
lr6	0.80974338	0.84392941	0.88331374	0.93050823	0.79281124	0.85	0.85606061	0.81100218	0.87777778	0.8817402
lr7	0.81546976	0.84629287	0.8832998	0.92958363	0.75190842	0.86601307	0.87103175	0.80119048	0.82857143	0.84217687
lr8	0.81742424	0.84108912	0.88536866	0.93230633	0.77526079	0.84166667	0.87053571	0.80151099	0.81875	0.83173077
lr9	0.82140216	0.84534764	0.88535354	0.93320366	0.76157076	0.89072039	0.85714286	0.81060606	0.85185185	0.77020202
lr10	0.81933962	0.84490183	0.89138776	0.92871417	0.76478257	0.85	0.8547619	0.82	0.78333333	0.85
svm2	0.80996169	0.85009488	0.9014121	0.93050542	0.80577283	0.925	0.85483871	0.85849057	0.74183007	0.85
svm3	0.821374	0.84108467	0.90141171	0.93050612	0.80586484	0.925	0.88730159	0.84920635	0.71212121	0.84997029
svm4	0.82899295	0.84629981	0.90141935	0.93050542	0.80650819	0.925	0.8875	0.85861823	0.76388889	0.85
svm5	0.82705128	0.84488298	0.90141414	0.93050834	0.80678389	0.925	0.87307692	0.84935065	0.8	0.87
svm6	0.82901185	0.84392402	0.90141052	0.93050823	0.80669201	0.91666667	0.88787879	0.85947712	0.73333333	0.8504902
svm7	0.83086229	0.84487061	0.90140845	0.93050485	0.8065079	0.91690009	0.88690476	0.84940476	0.8	0.85034014
svm8	0.83079837	0.84439992	0.90143369	0.93051428	0.80687587	0.91666667	0.88839286	0.8592033	0.8	0.85016026
svm9	0.83293003	0.84914429	0.901443	0.9305082	0.8066924	0.92490842	0.88888889	0.85016835	0.78703704	0.85016835
svm10	0.82888244	0.85011735	0.90142857	0.93051597	0.80660014	0.925	0.88571429	0.85909091	0.79166667	0.85
dta2	0.69877689	0.76043643	0.82899339	0.86371841	0.70502787	0.825	0.90322581	0.80188679	0.82843137	0.67
dta3	0.71976391	0.76139027	0.80293294	0.85111697	0.71541656	0.775	0.85396825	0.79338624	0.82575758	0.75965538
dta4	0.70441867	0.78225806	0.82904839	0.88628159	0.72230988	0.85	0.83854167	0.74501425	0.76736111	0.83
dta5	0.74663004	0.78559962	0.82505051	0.88628159	0.7270917	0.80833333	0.84230769	0.77445887	0.77142857	0.76
dta6	0.74059966	0.78418399	0.837129	0.90169898	0.72451584	0.83333333	0.87272727	0.72657952	0.82777778	0.78921569
dta7	0.75809524	0.78510767	0.82897384	0.88898177	0.71936893	0.84173669	0.83928571	0.75416667	0.8	0.74081633
dta8	0.76960956	0.76900781	0.83714158	0.90346419	0.72139184	0.85	0.84151786	0.76373626	0.825	0.72916667
dta9	0.73690932	0.78938999	0.81699134	0.89621762	0.71707716	0.86630037	0.88888889	0.75505051	0.75925926	0.80050505
dta10	0.73109579	0.75569849	0.83514286	0.89633088	0.71302754	0.80833333	0.85952381	0.75454545	0.825	0.77
rf2	0.76201739	0.81072106	0.89738794	0.91967509	0.78178283	0.925	0.88709677	0.81132075	0.76797386	0.78
rf3	0.7753084	0.8169026	0.88533528	0.92599917	0.78343827	0.91666667	0.9031746	0.78280423	0.82070707	0.83006536
rf4	0.77524956	0.82779886	0.89540323	0.92418773	0.78876824	0.89166667	0.92083333	0.80163818	0.81944444	0.83
rf5	0.78688645	0.81784174	0.89339394	0.9305491	0.7861029	0.90833333	0.89102564	0.82121212	0.82857143	0.83
rf6	0.77134011	0.81738054	0.88936233	0.92243929	0.78904348	0.90833333	0.92121212	0.83986928	0.81666667	0.85110294
rf7	0.80576577	0.82399884	0.88531187	0.93232454	0.78490756	0.90009337	0.92063492	0.83035714	0.85714286	0.82176871
rf8	0.80402098	0.81927461	0.88543267	0.92966062	0.788953	0.89166667	0.90625	0.84958791	0.85	0.86057692
rf9	0.80022854	0.82592997	0.89339827	0.93321823	0.78353132	0.92490842	0.9047619	0.82996633	0.81481481	0.84006734
rf10	0.78882438	0.82401715	0.89742857	0.92876331	0.78941256	0.9	0.88809524	0.82090909	0.83333333	0.81
b2	0.75631447	0.82163188	0.88933962	0.92509025	0.77681831	0.91666667	0.87096774	0.80188679	0.79738562	0.79
b3	0.79261622	0.80314795	0.88533528	0.92148978	0.78261053	0.89166667	0.90396825	0.79232804	0.79292929	0.83986928
b4	0.76754257	0.81783681	0.88730645	0.92148014	0.7832533	0.9	0.90416667	0.79166667	0.76388889	0.84
b5	0.79260073	0.81832468	0.89341414	0.92874322	0.78242672	0.88333333	0.92179487	0.83073593	0.8	0.84
b6	0.80210282	0.81309354	0.8813302	0.9323492	0.78766582	0.90833333	0.90454545	0.83986928	0.79444444	0.84987745
b7	0.81351351	0.82589728	0.89336016	0.93052191	0.78913557	0.90849673	0.9047619	0.83095238	0.85714286	0.81020408
b8	0.80789627	0.8197733	0.89340118	0.92964759	0.7876657	0.9	0.87276786	0.8489011	0.75625	0.84134615
b9	0.77922296	0.83021357	0.88336941	0.93504677	0.78518741	0.91697192	0.85449735	0.82070707	0.81481481	0.84006734
b10	0.78875181	0.83587678	0.89942857	0.9341769	0.78527602	0.90833333	0.88809524	0.83	0.86666667	0.83



Table XI:Summary of Stratified with K-Cross Validation with Machine Learning Algorithms

	kc2	kc1	cm1	pc1	jm1	ar1	ar3	ar4	ar5	ar6		Percentage
Linear Regression	0.8309	0.8534	0.89139	0.9332	0.79281	0.89072	0.8873	0.82063	0.87778	0.88174		30%
Suppport Vector Machine	0.8329	0.8501	0.90144	0.9305	0.80688	0.925	0.8889	0.85948	0.8	0.87		50%
Decision Tree	0.7696	0.7894	0.83714	0.9035	0.72709	0.8663	0.9032	0.80189	0.82843	0.83		0%
Random Forest	0.8058	0.8278	0.89743	0.9332	0.78941	0.925	0.9212	0.84959	0.85714	0.86058		0%
Bagging	0.8135	0.8359	0.89943	0.935	0.78914	0.91697	0.9218	0.8489	0.86667	0.84988		20%
												
Maximum	0.8329	0.8534	0.90144	0.935	0.80688	0.925	0.9218	0.85948	0.87778	0.88174		
Conclusion of Table XI:
Support Vector Machine algorithm achieved the highest performance in terms of most metrics, followed by Linear Regression and Bagging. Decision Tree and Random Forest performed relatively lower, with a percentage of 0%.
 
E) Neural Networks:

Abbreviations used in tables:
SPNN: Simple Perceptron Neural Network
MLNN: Multilayer Neural Network 
MLNN+P: Multilayer Neural Network + Permutation 

	Metric	SPNN	MLNN	MLNN+P
Kc2	Precision	0.64	0.85	0.7
	Recall	0.8	0.86	0.84
	F1-score	0.71	0.85	0.76
	Accuracy	0.8	0.86	0.84

	Metric	SPNN	MLNN	MLNN+P
Kc1	Precision	0.7	0.79	0.76
	Recall	0.83	0.83	0.61
	F1-score	0.76	0.79	0.66
	Accuracy	0.83	0.83	0.61
.
	Metric	SPNN	MLNN	MLNN+P
Cm1	Precision	0.9	0.96	0.88
	Recall	0.45	0.96	0.94
	F1-score	0.57	0.91	0.91
	Accuracy	0.45	0.96	0.94

	Metric	SPNN	MLNN	MLNN+P
Pc1	Precision	0.87	0.91	0.93
	Recall	0.93	0.89	0.82
	F1-score	0.9	0.9	0.86
	Accuracy	0.93	0.89	0.82


	Metric	SPNN	MLNN	MLNN+P
Jm1	Precision	0.72	0.78	0.79
	Recall	0.75	0.72	0.68
	F1-score	0.73	0.74	0.71
	Accuracy	0.75	0.72	0.68

	Metric	SPNN	MLNN	MLNN+P
Ar1	Precision	0.77	0.77	0.77
	Recall	0.88	0.88	0.88
	F1-score	0.82	0.82	0.82
	Accuracy	0.88	0.88	0.88

	Metric	SPNN	MLNN	MLNN+P
Ar3	Precision	0.85	0.85	0.84
	Recall	0.85	0.92	0.77
	F1-score	0.85	0.89	0.8
	Accuracy	0.85	0.92	0.77

	Metric	SPNN	MLNN	MLNN+P
Ar4	Precision	0.78	0.78	0.75
	Recall	0.41	0.68	0.86
	F1-score	0.48	0.72	0.8
	Accuracy	0.41	0.68	0.86

	Metric	SPNN	MLNN	MLNN+P
Ar5	Precision	0.48	0.51	0.9
	Recall	0.57	0.71	0.86
	F1-score	0.52	0.6	0.86
	Accuracy	0.57	0.71	0.86

	Metric	SPNN	MLNN	MLNN+P
Ar6	Precision	0.72	0.84	0.72
	Recall	0.85	0.75	0.85
	F1-score	0.78	0.78	0.78
	Accuracy	0.85	0.75	0.85

Conclusion of Neural Network:
In summary, the MLNN model generally outperformed the other models across various metrics, achieving higher precision, recall, F1-score, and accuracy. The MLNN+P model showed mixed performance, with varying results across different metrics. The SPNN model had lower performance compared to MLNN and MLNN+P in most cases.

F. Experimental Results
We performed experiments on the ten different datasets
which have a different number of attributes and results were shown in Tables II to XI. There are many ways to evaluate any machine learning algorithm and evaluation of the model is a very essential part of any project. In this experiment, different evaluation metrics which are given above are used to evaluate model performance. For each machine learning algorithm and each dataset, the best classification performance result is highlighted in Yellow. The first notable observation from these experimental results which are shown in Table III is that Random Forest, Decision Tree and Bagging learning algorithm which is a tree-based algorithm is better than other learning algorithm categories. The performance difference between each machine learning algorithm is shown in above tables clearly. As shown in above,the results of the tree-based learning algorithm are better compared to other algorithms except for KC2 dataset. Although datasets of different sizes were used, no major differences were observed in performance. Table II, IV, VI, VIII, IX, XI shows that ensemble learners are better at software defect estimation and it is also a powerful way to improve the performance of the model. It
is a more successful model than individual models because of combining several diverse classifiers together. In neural network SPNN perform better than other neural network in PC1, JM1 while MLNN performed in KC2, KC1, CM1, AR3 and MLNN+P performed in AR4, AR5,AR6. All neural network were trained in epochs = 1000

IV. CONCLUSIONS

In this experimental study, five machine learning algorithms and 3 neural network are used to predict defectiveness of software systems before they are released to the real environment and/or delivered to the customers and the best category which has the most capability to predict the software defects are tried to find while comparing them based on software quality metrics
which are accuracy, precision, recall and F-measure. We
carry out this experimental study with five NASA datasets
which are JM1, PC1, CM1, KC1 and KC2 and five Softlab datasets AR1, AR3, AR4, AR5, AR6 . These datasets are
obtained from public PROMISE repository and other open sources. The results of this experimental study indicate that tree-structured classifiers in other words ensemble learners which are Random Forests and Bagging have better defect prediction performance compared to its counterparts. Especially, the capability of Bagging in predicting software defectiveness is better. When applied to all datasets, the overall accuracy, precision, recall and FMeasure
of Bagging is within 83.7-94.1%, 81.3-93.1%, 83.7-
94.1% and 82.4-92.8% respectively. For PC1 dataset, Bagging outperforms(94.5%) all other machine learning techniques in all quality metric. However, Bagging outperforms it in accuracy and recall for CM1 dataset. Random Forests outperforms all machine learning techniques in all quality metrics for KC1 dataset. Finally, for KC2 dataset. It is deductive from obtained results
that tree-structured classifiers are more suitable for software defect prediction. Moreover, it is recommended to software companies to utilize tree-structured classifiers for software defect prediction due to its performance. Utilizing these techniques enables them to save software testing and maintenance costs by identifying defects in the early phase of project life cycle and taking corrective and preventive actions before they becomes failures. 


V. FUTURE WORK

Conducting additional experimental studies by using different datasets would be one direction of future work. These datasets would be obtained from the open repositories or software companies. Second direction of the future work would be conducting an experimental study by applying deep learning(genetic) algorithms additional to these machine learning algorithms. Bringing into existence of new attributes by using combination of previous attributes would be another direction of the future work. In conclusion, it would be practical to carry out a case study by using distinct software quality datasets obtained from real-life projects of software companies having different company sizes.

VI. KEYWORDS

SPNN: Simple Perceptron Neural Network
MLNN: Multilayer Neural Network 
MLNN+P: Multilayer Neural Network + Permutation
lr: Linear Regression
svm: Support Vector Machine
dt:Decision Tree
rf: random forest
b: bagging
lr2: here lr is Linear regression and the number Two represents the value of k in k-fold.
svm2: here svm is Support Vector Machine and the number Two represents the value of k in k-fold.
dt2: here dt is Decision Tree and the number Two represents the value of k in k-fold.
rf2: here rf is random forest and the number Two represents the value of k in k-fold.
b2 : here b is Bagging and the number two represents the value of k in k fold.


