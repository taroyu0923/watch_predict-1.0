from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np

'''
# method to transform csv file to dataframe

input: path of data(.csv file)
output dataframe and print head of dataset
'''
def Read_from_csv(path):
    test_array = pd.read_csv(path)
    print(test_array.head)
    return test_array


def Set_Decision_Tree(X1_train, X1_test, y1_train, y1_test):
    decision_Tree_Model = DecisionTreeClassifier()
    decision_Tree_Model.fit(X1_train, y1_train)
    decision_Tree_Model_result = decision_Tree_Model.predict(X1_test)
    decision_Tree_report = classification_report(y1_test, decision_Tree_Model_result)
    decision_Tree_matrix = confusion_matrix(y1_test, decision_Tree_Model_result)
    decision_Tree_score = accuracy_score(y1_test, decision_Tree_Model_result)
    print(decision_Tree_report, "\n", decision_Tree_matrix, "\n", decision_Tree_score)
    return decision_Tree_Model


def Set_Random_Forest(X1_train, X1_test, y1_train, y1_test):
    random_forest_Model = RandomForestClassifier(criterion= 'entropy', n_estimators= 50, random_state= 3, n_jobs = 2)
    random_forest_Model.fit(X1_train, y1_train)
    random_forest_Model_result = random_forest_Model.predict(X1_test)
    random_forest_report = classification_report(y1_test, random_forest_Model_result)
    random_forest_matrix = confusion_matrix(y1_test, random_forest_Model_result)
    random_forest_score = accuracy_score(y1_test, random_forest_Model_result)
    print(random_forest_report, "\n", random_forest_matrix, "\n", random_forest_score)
    return random_forest_Model


# main
def main():

    try:

        # data input
        csv_file_path = 'D:/WORK4/watch_predict-1.0/input_data/test_1.csv' # you need to add data's path here (csv file)
        res = Read_from_csv(csv_file_path)

        # new dataset
        X = res[['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']]
        y = res[['activity']] 


        # test dataset
        '''
        path2 = 'D:/WORK4/watch_predict-1.0/input_data/0_Accelerometer_Sun-Apr-28-11_57_07-GMT08_00-2019_Log.csv'
        res_test = Read_from_csv(path2)
        X2 = res_test[['acceleration_x', 'acceleration_y', 'acceleration_z']]
        print(X2.head())
        '''

        y = np.ravel(y)     # change to array
        print(X.head(), '\n', 'y=', y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # test data(Decision Tree / Random forest)
        model_For_Decision_Tree = Set_Decision_Tree(X_train, X_test, y_train, y_test)
        model_For_Random_forest = Set_Random_Forest(X_train, X_test, y_train, y_test)

    except Exception as e:

        print('exception type is:')
        print(type(e), str(e))
        print('exception happened')


    joblib.dump(model_For_Decision_Tree, 'saved_model/md1_Dec_Tree.pkl')
    joblib.dump(model_For_Random_forest, 'saved_model/md2_Ram_For.pkl')
    print('dump finish!')


if __name__ == "__main__":
    main()