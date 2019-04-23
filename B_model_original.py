'''

This is original script
Divide to "predict_main.py" & "buildmodel_main.py"

'''

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# import seaborn as sns
# import matplotlib.pyplot as plt
# import csv 

# read file from csv
def Read_from_csv(path):
    try:
        test_array = pd.read_csv(path)
        print(test_array.head)
    except Exception as e:
        print('exception type is:')
        print(type(e), str(e))
        print('exception happened')
    return test_array

def Change_to_CSV(path):
    data_txt = np.loadtxt(path)
    data_txtDF = pd.DataFrame(data_txt)
    data_txtDF.to_csv('result_change.csv', index=False)


def Set_Decision_Tree(X_train, y_train):
    decision_Tree_Model = DecisionTreeClassifier()
    decision_Tree_Model.fit(X_train, y_train)
    return decision_Tree_Model


def Predict_Decision_Tree_Model(model, data):
    result = model.predict(data)
    return result

def Set_Random_Forest(X_train, y_train):
    random_forest_Model = RandomForestClassifier(criterion='entropy', n_estimators=50, random_state=3, n_jobs =2)
    random_forest_Model.fit(X_train, y_train)
    return random_forest_Model

def Predict_Random_Forest_Model(model, data):
    result = model.predict(data)
    return result

# main

def main():
    # data input

    csv_file_path = 'D:/WORK4/PROJECT/test_1.csv' # you need to add data's path here (csv file)
    #txt_file_path = 'D:/WORK4/PROJECT/0_Accelerometer_Mon-Apr-22-13_27_16-GMT08_00-2019_Log.txt'
    res = Read_from_csv(csv_file_path)
    #res1 = Change_to_CSV(txt_file_path)

    # new dataset

    X = res[['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    print(X.head())
    y = res[['activity']] 
    y = np.ravel(y)
    print("y=" , y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print(y_train)

    # test data(Decision Tree / Random forest)

    try:

        model_For_Decision_Tree = Set_Decision_Tree(X_train, y_train)
        model_For_Random_forest = Set_Random_Forest(X_train, y_train)

        predct_result_1 = Predict_Decision_Tree_Model(model_For_Decision_Tree, X_test)
        predct_result_2 = Predict_Random_Forest_Model(model_For_Random_forest, X_test)
    except Exception as e:

        print('exception type is:')
        print(type(e), str(e))
        print('exception happened')

    joblib.dump(model_For_Decision_Tree, 'saved_model/md1_Dec_Tree.pkl')
    joblib.dump(model_For_Random_forest, 'saved_model/md2_Ram_For.pkl')
    print(predct_result_1, predct_result_2)


if __name__ == "__main__":
    main()