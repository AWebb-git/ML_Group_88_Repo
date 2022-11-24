import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn 
from statistics import mean
from sklearn.metrics import mean_squared_error,r2_score,f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
from sklearn.model_selection import KFold
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report

def getF1ScoreInfo(X, Y, K, F1ScoreKNN, F1ScoreKNNSTD):
    #make the model
    KNNmodel = KNeighborsClassifier(n_neighbors=K,weights='uniform')
    f1Holder=[]
    kf = KFold(n_splits=5)
    #split the data into training and testing 
    for train, test in kf.split(X):
        #fit model, get preidtcion and compare preidctions with actual predictions
        KNNmodel.fit(X[train], Y[train])
        KNNPrediction = KNNmodel.predict(X[test])
        f1Holder.append(f1_score(Y[test],KNNPrediction, average = "micro"))
        #F1 score but only in a 1d array as only one parameter
    averageF1 = mean(f1Holder)
    F1ScoreKNN.append(averageF1)
    F1ScoreKNNSTD.append(np.array(f1Holder).std())

    return F1ScoreKNN, F1ScoreKNNSTD


def plotCrossVal(polyOrder, k, F1ScoreKNNPoly, F1ScoreKNNSTDPoly):
    theLabels = []
    print(polyOrder)
    for i in range(7):
        theLabels.append("Ploy = %d"%(polyOrder[i]))
    #plot the F1 score for all C and poly values
    for i in range(7):
        plt.errorbar(k,F1ScoreKNNPoly[i,:],yerr=F1ScoreKNNSTDPoly[i,:], label=theLabels[i])
        #plt.plot(C,F1ScoreLogisticPoly[i,:], label=theLabels[i])
    plt.title("F1 Cross-Validation Plot for KNN")
    plt.xlabel('K')
    plt.ylabel('F1 Score')
    plt.xlim((0,60))
    plt.legend()
    plt.show()


def workOutHyperparameters(X,Y,size):
    polyOrder = [1,2,3,4,5,10,25]
    k = [1,3,5,7,9,11,13,15,17,19,21,31,41,51,61,71,81,91,101]
    inital = 1
    F1ScoreKNN=[]
    F1ScoreKNNPoly=[]

    F1ScoreKNNSTD=[]
    F1ScoreKNNSTDPoly=[]

    for po in polyOrder:
        Xpoly = PolynomialFeatures(po).fit_transform(X)
        F1ScoreKNN=[]
        F1ScoreKNNSTD=[]
        for Ki in k:
            F1ScoreKNN, F1ScoreKNNSTD = getF1ScoreInfo(Xpoly, Y, Ki, F1ScoreKNN, F1ScoreKNNSTD)
        if(inital == 1):
            #this stores all the F1 scores for every C value, for every poly order
            #this initalises the 2d array
            #row being poly order
            #column being C value
            F1ScoreKNNPoly = F1ScoreKNN
            F1ScoreKNNSTDPoly = F1ScoreKNNSTD
            inital = 2 
        else:
            #and then stack each row element into the F1 array
            F1ScoreKNNPoly = np.vstack((F1ScoreKNNPoly, F1ScoreKNN))
            F1ScoreKNNSTDPoly = np.vstack((F1ScoreKNNSTDPoly, F1ScoreKNNSTD))

        F1ScoreKNN=[]
        F1ScoreKNNSTD=[]
        #this is the end of the penality for loop
    plotCrossVal(polyOrder, k, F1ScoreKNNPoly, F1ScoreKNNSTDPoly)


def main():

    df = pd.read_csv("formatted_data.csv")
    #X1 = df.loc[:,"review_count"]
    #X2 = df.loc[:,"rating"]
    #X3 = df.loc[:,"price"]
    X1 = df.iloc[:,2]
    X2 = df.iloc[:,3]
    X3 = df.iloc[:,4]
    X = np.column_stack((X1,X2,X3))
    #Y = df.iloc[:,"user_rating"]
    Y = df.iloc[:,1]
    #train_user_mean_rating = df["user_rating"].mean()
    size = len(Y)
    workOutHyperparameters(X,Y,size)


    #now with selected parameters
    #selectedK = 1
    #selectedPolyOrder = 1
    #Xpoly = PolynomialFeatures(selectedPolyOrder).fit_transform(X)


    #SelectedKNNmodel = KNeighborsClassifier(n_neighbors=selectedK,weights='uniform').fit(Xpoly, Y)
    #KNNPrediction = SelectedKNNmodel.predict(Xpoly)
    #y_frequentPred = np.argmax(KNNPrediction, axis=1)
    #y_trainFreq = np.argmax(Y, axis=1)
    #print(classification_report(y_trainFreq, y_frequentPred))
    #print(confusion_matrix(y_trainFreq,y_frequentPred))


if __name__ == "__main__":
    main()