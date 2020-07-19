\

IMPLEMENTATIONOF

KNN,HBOSANDLOF

ALGORITHMS

USINGHTTPREQUESTDATASET

**GroupMembers**

RaloueRKapoor

AkshitiParashar

AnshTiwari

SrihariAnanthan



\

INDEX

**1. INTRODUCTION**

**2. COMMONCODE**

\1. IMPORTINGLIBRARIES

\2. FUNCTIONFORDATAPREPROCESSING

\3. FUNCTIONFORCONVERTINGCLASSES

TONUM(-1,0,1)

\4. FUNCTIONFORGETTINGOPTIMAL

HYPERPARAMETERS

\5. FUNCTIONFORPRINTINGRESULTS

**3. THEPROBLEMSTATEMENT**

**4. KNN**

**5. HBOS**

**6. LOF**

\1. WITHOUTNOVELTY

\2. WITHNOVELTY



\

INTRODUCTION

Thisisarecapofthealgorithmswehavelearnttillnow,thatareusedfor

unsupervisedAnomalydetection-thatis,tofindoutoutliersindatasetswithout

anypriortraining.Weshallbeexploringthreealgorithms-K-NearestNeighbors,

HistogramBasedOutlierScoresandLocalOutlierFactor.Asassigned,wehave

usedtheHTTPdatasetforourwork,theresultsofwhichhavebeenattachedfor

betterunderstandingofhowthesealgorithmswork,alongwithdueexplanationof

thecode.



\

2.1 IMPORTING LIBRARIES

**NumPy **provides n-

dimensional array object.

Also provides

importnumpyasnp

importpandasaspd

importmatplotlib.pyplotasplt

mathematical functions

which can be used in

many calculations.

fromsklearn.preprocessingimportStandardScaler

fromsklearn.model_selectionimporttrain_test_split

fromsklearn.neighborsimportKNeighborsClassifier,LocalOutlierFactor

fromsklearn.metricsimportconfusion_matrix,classification_report,roc_curve,auc

frompyod.models.hbosimportHBOS

**Matplotlib**-

scientific

plotting library

usually

required to

visualize data.

**PyOD**-Python

**Scikit-learn**-

**Pandas**- used for

data analysis. It can

take multi-

dimensional arrays

as input and

Outlierdetection-for

anomalydetection

modelingwith

supervisedand

unsupervised

provides tools for

data analysis and

data mining. It

provides

classification and

clustering algorithms.

produce

charts/graphs.

learningtechniques



\

2.2FUNCTIONFORDATAPREPROCESSING

defdata_preprocessing(input_data):

df=pd.read_csv('/content/drive/MyDrive/Com-Olho-Datasets/'+input_data+'.csv',header=None)

l=df.columns.values.tolist()

lastcolname=l[-1]

Loadingthedataintothedataframe

df=df.rename(columns={lastcolname:'TV'})

datay=df['TV']

datax=df.drop(['TV'],axis=1)

Renaminglastcolumnas‘TV’(Test

Vector)

scaler=StandardScaler()

scaler.fit(datax)

datax=scaler.transform(datax)

returndatax,datay

Dividingthedatainto2partsfordata

standardization

Standardize/Cleaningdata



\

2.3FUNCTIONFORCONVERTINGCATEGORICALCLASSESTOBINARY

defconvert_to_num(datay,algotype):

ifalgotype=='KNN'oralgotype=='HBOS':

datay=datay.replace({'n':0,'o':1})

elifalgotype=='LOF_no_novelty'oralgotype=='LOF_novelty':

datay=datay.replace({'n':1,'o':-1})

returndatay

n:Normal

o:Outlier



\

2.4FUNCTIONFORFINDINGHYPERPARAMETERS

deffindhyperparameter(algotype,xtrain,ytrain,xtest,ytest):

error_rate=[]

hplist=[]

ifalgotypeis'KNN':

foriinrange(1,50):

print("Checkingk="+str(i))

neigh=KNeighborsClassifier(n_neighbors=i)

neigh.fit(xtrain,ytrain)

Hyperparameters:

● Typeofalgorithm

● Trainingset

● Testingset

pred=neigh.predict(xtest)

error_rate.append(np.mean(pred!=ytest))

elifalgotypeis'LOF_no_novelty':

foriinrange(10,20):

print("Checkingk="+str(i))

clf=LocalOutlierFactor(n_neighbors=i)

pred=clf.fit_predict(xtrain)

error_rate.append(np.mean(pred!=ytrain))

elifalgotypeis'LOF_novelty':

foriinrange(10,20):

Findingminimumerrorbypassing

differentvaluesofnearestneighbors(k)

print("Checkingk="+str(i))

clf=LocalOutlierFactor(n_neighbors=i,novelty=True)



\

clf.fit(xtrain)

pred=clf.predict(xtest)

error_rate.append(np.mean(pred!=ytest))

elifalgotypeis'HBOS':

foriinrange(10,100):

print("Checkingn_bins="+str(i))

clf=HBOS(n_bins=i)

clf.fit(xtrain)

Findingminimumerrorbypassing

differentvaluesofbins(n)

pred=clf.predict(xtest)

error_rate.append(np.mean(pred!=ytest))

foriinrange(10,20):

hplist.append(i)

Gettingindexwhereerrorrateis

minimum

min_val=min(error_rate)

ind=np.where(error_rate==min_val)

ind=ind[0][0]

returnhplist[ind],hplist,error_rate



\

2.5FUNCTIONFORPRINTINGRESULTS

defprint_results(optimal_value,hpvalues,error_rate,y_test,preds,dataset,algotype):

print("Thisisananalysisfor"+algotype+"on"+dataset+"dataset")

print("Optimalvalueis"+str(optimal_value))

plt.plot(hpvalues,error_rate)

plt.title('errorratevshpvalues')

plt.xlabel('hpvalues')

hpvalues:Hyperparametervalues

plt.ylabel('errorrate')

plt.show()

cm=confusion_matrix(y_test,preds)

cr=classification_report(y_test,preds)

print("Printingconfusionmatrix")

print(cm)

print("Printingclassificationreport")

print(cr)

print("PrintingROC_AUCCurve")

fpr,tpr,threshold=roc_curve(y_test,preds)

roc_auc=auc(fpr,tpr)

AUC:AreaUnderCurve

ROC:

curve

ReceiverOperatingCharacteristics

plt.plot(fpr,tpr,roc_auc)

plt.show()



\

3.PROBLEMSTATEMENT-

THEHTTPREQUESTDATASET

X:INDEPENDENTVARIABLES

Y:DEPENDENTVARIABLE



\

4.K-NearestNeighbors

**K-Nearest Neighbors **is an algorithm for supervised learning. Where the data is 'trained' with data points corresponding to

their classification. Once a point is to be predicted, it takes into account the 'K' nearest points to it to determine its

classification.



\

Euclideandistanceformulaformulti-dimensionalvectors**:**

● **KNNALGORITHM**

\1. Importunsupervised.csvfileandconvert

ittosupervised.

\1. Convertcategorical classes into binary.

\2. Do standardization on independent

variables.

\1. Perform train test split.

\2. Do classification..

\3. Plot accuracy and filter out best K.

\4. Plot error rate curve

\5. Predict the model.

\6. Get confusion matrix and compute

classification report for best K.

\7. Get ROC curve and find AUC.



\

**defKNN_algorithm(dataset):**

**Callingthefunction**

**Datapreprocessing**

**datax,datay=data_preprocessing(dataset)**

print("DataPreprocessed")

**Convertingcategorical classes into binary.**

**datay=convert_to_num(datay,'KNN')**

**x_train,x_test,y_train,y_test=train_test_split(datax,datay,test_size=0.3,random_state=4)**

**optimal_value,hpvalues,error_rate=findhyperparameter('KNN',x_train,y_train,x_test,y_test)**

print("Optimalvaluefound")

**Putting**

**optimalvalue**

**asK**

**TRAINTESTSPLIT**

**ANDFINDING**

**OPTIMALK**

**neigh=KNeighborsClassifier(n_neighbors=optimal_value)**

**neigh.fit(x_train,y_train)**

**Fittingandpredictingthe**

**xandytraindata**

**preds=neigh.predict(x_test)**

**Plottingresults**

**print_results(optimal_value,hpvalues,error_rate,y_test,preds,dataset,'KNN')**



\

● **Computationofaccuracyusingconfusionmatrix:**

TP=18570 FP=0

**Predictionconductionpositive=TP+FP**

**=18750**

**Predictionconductionnegative=FN+TN**

**=33**

FN=1

TN=32

**ACC=(18570+32)/(18570+32+0+1)=0.999=99.99%**



\

● **Classificationreport**

**recall**

**1.00**

**0.97**

**f1-score**

**1.00**

**precision**

**1.00**

**0**

**1**

**1.00**

**0.98**

**accuracy**

**macro avg**

**1.00**

**1.00**

**0.98**

**1.00**

**0.99**

**weighted avg 1.00**

**1.00**



\

● **Results**

**WhatisthevalueoftheAUROC**

**(areaunderroc)toconcludethata**

**classifierisexcellent?**



\

5.HistogramBasedOutlierScore

Standardizedxtraindata

Column A

0

1

2

3

4

0.00

0.31

0.90

0.45

0.50

nbins=3

Similarlywefind

histogramsfor

othercolumnsas

well

Min: 0 Max: 0.9



\

Afterwehaveallhistogramsreadywestartbringing

invaluestocalculatethescoresforthem

Supposewecalculateforx=0.59

Weapplythisformulaforcalculatingthescores

0.59

dstandsforthenumberoffeaturesorcolumns

pistheinputtestdatawhichneedstobe

checked

histpisthefrequencyforthepointpina

histogram

HBOS(0.59)=log(⅓).

i

Finallywegetascoretocheckforoutlier



\

HBOSFORREQUESTSDATA

TRAINTESTSPLITAND

FINDINGOPTIMAL

NBINS

DATAPREPROCESSING

**def HBOS_algorithm(dataset):**

**datax,datay = data_preprocessing(dataset)**

**datay = convert_to_num(datay,'HBOS')**

**x_train,x_test,y_train,y_test = train_test_split(datax,datay,test_size=0.3,random_state=4)**

**optimal_value,hpvalues,error_rate = findhyperparameter('HBOS',x_train,y_train,x_test,y_test)**

**clf = HBOS(n_bins=optimal_value)**

**clf.fit(x_train)**

**preds = clf.predict(x_test)**

**print_results(optimal_value,hpvalues,error_rate,y_test,preds,dataset,'HBOS')**

PRINTINGRESULTS

TRAININGANDPREDICTING



\

RESULT

S

ConfusionMatrix

ClassificationReport

TP = 16726

FN = 5

FP = 1844

TN = 28

precision

recall

f1-score

support

0

1

1.00

0.01

\-

0.90

0.85

\-

0.95

0.03

0.90

0.49

18570

33

Accuracy=(TP+TN)

accuracy

18603

18603


\_________________

macro

avg

0.51

0.87

(TP+TN+FP+FN)

Accur ac y=90.06%

weighted

avg

1.00

0.90

0.95

18603



\

Errorratevsnbins

Roc_curve

AUC=0.87

Falsepositiverate

Optimalbinswasfoundatnbins=61



\

6.LocalOutlierFactor

**Local Outlier Factor (LOF) **is a score that tells how likely a certain data point is an outlier or

anomaly.

**CONCEPT: **Local Density

**LOCALITY: ***k *nearest neighbors *(Distance=>Density)*

**OUTLIERS: **Points that have a substantially lower density than their neighbors

**MATHEMATICAL CALCULATION:**

a)Look at the neighbors of a certain point

b) Find out its density

c) Compare this to the density of other points later



\

**K-distance: **Is the distance of a point to its *kth *neighbor. If *k*

was 3, the *k-distance *would be the distance of a point to the

third closest point.

**Reachability distance: **The k-distance is now used

calculate the reachability distance. This distance measure is

the maximum of the distance of two points and the k-distance

of the second point.

**reach-dist(a,b) = max{k-distance(b), dist(a,b)}**



\

**Local reachability density: **The *reach-dist *is then used to calculate the lrd.

Basically, the local reachability density tells how far we have to travel from our point to reach the next

point or cluster of points.

**lrd(a) = 1/(sum(reach-dist(a,n))/k)**

The lrd of each point will then be compared to the lrd of their *k *neighbors. The LOF is basically the

average ratio of the lrds of the neighbors of *a *to the lrd of *a*.

**LOF ≈1 **⇒ **no outlier**

**LOF **≫**1 **⇒ **outlier**



\

**Outlier detection**

**Novelty detection**

● Observations that are far from the others.

● We are interested in detecting whether a

**new **observation is an outlier.

● Outlier detection estimators try to fit the

regions where the training data is the

most concentrated.

● Semi-supervisedanomalydetection.

● Novelties/anomaliescanformadense

clusteraslongastheyareinalowdensity

regionofthetrainingdata,consideredas

normalinthiscontext.

● Unsupervisedanomalydetection

● Outliers/anomaliescannotformadense

clusterasavailableestimatorsassumethat

theoutliers/anomaliesarelocatedinlow

densityregions.



\

6.1LOFFORREQUESTSDATA

(withoutnovelty)

DATAPREPROCESSING

TRAININGANDPREDICTING

**def LOF_no_novelty_algorithm(dataset):**

**datax,datay = data_preprocessing(dataset)**

**print("Data Preprocessed")**

**datay = convert_to_num(datay,'LOF_no_novelty')**

**x_train = datax**

**y_train = datay**

**x_test = None**

**y_test = None**

**optimal_value,hpvalues,error_rate = findhyperparameter('LOF_no_novelty',x_train,y_train,x_test,y_test)**

**clf = LocalOutlierFactor(n_neighbors=optimal_value)**

**preds = clf.fit_predict(x_train)**

**print_results(optimal_value,hpvalues,error_rate,y_train,preds,dataset,'LOF_no_novelty')**

Dataalteringandassignment

Printingresults



\

DATAPREPROCESSING

6.2LOFFORREQUESTSDATA

(withnovelty)

**def LOF_novelty_algorithm(dataset):**

**datax,datay = data_preprocessing(dataset)**

**datay = convert_to_num(datay,'LOF_novelty')**

**x_train,x_test,y_train,y_test = train_test_split(datax,datay,test_size=0.3,random_state=4)**

**ytrainlist = []**

**for i in y_train:**

**ytrainlist.append(i)**

**j = 0**

**xtrainlist = []**

**for i in x_train**

**m = []**

TestTrainsplit,dataappend

andsorting

**m = i.tolist()**

**m.append(ytrainlist[j])**

**xtrainlist.append(m)**

**j = j + 1**

TRAININGANDPREDICTING

**df_rejoined = pd.DataFrame(xtrainlist)**

**df_rejoined = df_rejoined[df_rejoined[30]==1]**

**new_y_train = df_rejoined[30]**

**new_x_train = df_rejoined.drop([30],axis=1)**

**optimal_value,hpvalues,error_rate=findhyperparameter('LOF_novelty',new_x_train,new_y_train,x_test,y_test**

**)**

Printingresults

**clf = LocalOutlierFactor(n_neighbors=optimal_value,novelty=True)**

**clf.fit(new_x_train)**

**preds = clf.predict(x_test)**

**print_results(optimal_value,hpvalues,error_rate,y_test,preds,dataset,'LOF_novelty')**



\

RESULTS

**LOF with Novelty**

**LOF without Novelty**

ConfusionMatrix

ConfusionMatrix

33

0

19

86

879

17691

2663

59242

ClassificationReport

ClassificationReport

Precision

Recall

f1-score

0.07

support

Precision

Recall

f1-score

support

-1

1

0.04

1.00

\-

1.00

0.95

\-

33

-1

1

0.01

1.00

\-

0.18

0.96

\-

0.01

0.98

0.96

105

0.98

0.95

18570

18570

61905

62010

accurac

y

accurac

y

macro

avg

0.52

1.00

0.98

0.95

0.52

0.97

18570

18570

macro

avg

0.50

1.00

0.57

0.96

0.50

0.98

62010

62010

weighted

avg

weighted

avg



\

**LOF with Novelty**

**LOF without Novelty**

ErrorratevsHyperparametervalues

ErrorratevsHyperparametervalues

Optimal value is 10

Optimal value is 11



\

**LOF with Novelty**

**LOF without Novelty**

Falsepositiverate

Falsepositiverate

Accuracy:95.27%

Accuracy:95.56%



\

THANKYOU


