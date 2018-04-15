
import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
train.drop('id', axis=1, inplace=True)
cleanup_nums = {"gender"        : {"Female": 0, "Male": 1, "Other": -1},
                "ever_married"  : {"No": 0, "Yes": 6},
                "work_type"     : {"children":0, "Govt_job": 1, "Never_worked": 2, "Self-employed": 3, "Private": 4},
                "Residence_type": {"Rural": 0, "Urban":1},
                "smoking_status": {"never smoked": 0, "formerly smoked": 1, "smokes": 2}
               }
train.replace(cleanup_nums, inplace=True)

'''train_child = train.copy()
A = ((train["work_type"]==0) & ((train["smoking_status"]==2) | (train["smoking_status"]==1)))
True in A
A.index[A == True].tolist()
B = ((train["work_type"]==0) & (train["smoking_status"]==2))
C = ((train["work_type"]==0) & (train["smoking_status"]==1))
D = ((train["work_type"]==0) & (train["smoking_status"]==0))
True in B
B = B.index[B == True].tolist()
True in C
C = C.index[C == True].tolist()'''

train.bmi.fillna(28.605, inplace=True)

for i in range(len(train["work_type"])):
    if ((train["work_type"][i]==0) and (train["smoking_status"][i]==np.nan)):
        train["smoking_status"][i]=0
train.smoking_status.fillna(1, inplace=True)

train.age[(train["age"]>=0.0) & (train["age"]<=16.0)]=0.0
train.age[(train["age"]>16.0) & (train["age"]<=35.0)]=1.0
train.age[(train["age"]>35.0) & (train["age"]<=50.0)]=2.0
train.age[(train["age"]>50.0)]=3.0

train.bmi[(train["bmi"]>=0.0) & (train["bmi"]<=18.5)]=0.0
train.bmi[(train["bmi"]>18.5) & (train["bmi"]<=24.9)]=1.0
train.bmi[(train["bmi"]>24.9) & (train["bmi"]<=29.9)]=2.0
train.bmi[(train["bmi"]>29.9)]=3.0

train.avg_glucose_level[(train["avg_glucose_level"]>=0.0) & (train["avg_glucose_level"]<=65.0)]=0.0
train.avg_glucose_level[(train["avg_glucose_level"]>65.0) & (train["avg_glucose_level"]<=150.0)]=1.0
train.avg_glucose_level[(train["avg_glucose_level"]>150.0) & (train["avg_glucose_level"]<=240.0)]=2.0
train.avg_glucose_level[(train["avg_glucose_level"]>240.0)]=3.0
#------------ML analysis-------------------
from sklearn.cross_validation import train_test_split

features = train.iloc[:,0:10]
labels = train['stroke']
x1,x2,y1,y2 =train_test_split(features, labels, random_state=0, train_size =0.3)


        
#Classifier imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Performance metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
        
gnb = GaussianNB()

# Train our classifier and test predict
gnb.fit(x1, y1)
y2_GNB_model = gnb.predict(x2)
print("GaussianNB Accuracy :", accuracy_score(y2, y2_GNB_model))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y2, y2_GNB_model)
print("GNB_auc_score", auc(false_positive_rate, true_positive_rate))
print("GNB_roc_auc_score", roc_auc_score(y2, y2_GNB_model))




#-----------------test------------------------

test = pd.read_csv("test.csv")
test.drop('id', axis=1, inplace=True)
cleanup_nums = {"gender"        : {"Female": 0, "Male": 1, "Other": -1},
                "ever_married"  : {"No": 0, "Yes": 6},
                "work_type"     : {"children":0, "Govt_job": 1, "Never_worked": 2, "Self-employed": 3, "Private": 4},
                "Residence_type": {"Rural": 0, "Urban":1},
                "smoking_status": {"never smoked": 0, "formerly smoked": 1, "smokes": 2}
               }
test.replace(cleanup_nums, inplace=True)


test.bmi.fillna(28.545, inplace=True)

for i in range(len(test["work_type"])):
    if ((test["work_type"][i]==0) and (test["smoking_status"][i]==np.nan)):
        test["smoking_status"][i]=0
test.smoking_status.fillna(1, inplace=True)

test.age[(test["age"]>=0.0) & (test["age"]<=16.0)]=0.0
test.age[(test["age"]>16.0) & (test["age"]<=35.0)]=1.0
test.age[(test["age"]>35.0) & (test["age"]<=50.0)]=2.0
test.age[(test["age"]>50.0)]=3.0

test.bmi[(test["bmi"]>=0.0) & (test["bmi"]<=18.5)]=0.0
test.bmi[(test["bmi"]>18.5) & (test["bmi"]<=24.9)]=1.0
test.bmi[(test["bmi"]>24.9) & (test["bmi"]<=29.9)]=2.0
test.bmi[(test["bmi"]>29.9)]=3.0

test.avg_glucose_level[(test["avg_glucose_level"]>=0.0) & (test["avg_glucose_level"]<=65.0)]=0.0
test.avg_glucose_level[(test["avg_glucose_level"]>65.0) & (test["avg_glucose_level"]<=150.0)]=1.0
test.avg_glucose_level[(test["avg_glucose_level"]>150.0) & (test["avg_glucose_level"]<=240.0)]=2.0
test.avg_glucose_level[(test["avg_glucose_level"]>240.0)]=3.0


x_test = test.iloc[:,0:10]
y_test_KNN_model = gnb.predict(x_test)

df = pd.DataFrame(y_test_KNN_model)
df.to_csv('out.csv')