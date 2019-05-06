from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np

'''


Parameters
--------

Read_from_csv: string, optional(file's path, only for csv file)
    Read the csv file and put them into dataframe

Set_Decision_Tree: Double array * 4(X training data, X testing data, y training data, y testing data)
    Input the data which want to training,
    This method will train data by decision tree, then return model
    Finally, it would print out test score of return model, the score would be:
        classification report
        Confusion matrix
        Accuracy score

Set_Random_Forest:Double array * 4(X training data, X testing data, y training data, y testing data)
    Input the data which want to training,
    This method will train data by random forest, then return model
    Finally, it would print out test score of return model, the score would be:
        classification report
        Confusion matrix
        Accuracy score


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
        acce_file_path = 'D:/Year_PROJ/watch_predict-1.0/input_data/Accelerometer_Merge_Log.csv' # you need to add acceleration data's path here (csv file)
        gyro_file_path = 'D:/Year_PROJ/watch_predict-1.0/input_data/Gyro_Merge_Log.csv' # you need to add Gyroscope data's path here (csv file)
        acce_res = Read_from_csv(acce_file_path)
        gyro_res = Read_from_csv(gyro_file_path)

        # new dataset
        acce_X = acce_res[['acceleration_x', 'acceleration_y', 'acceleration_z']]
        acce_y = acce_res[['activity']] 
        acce_y = np.ravel(acce_y)     # change to array
        print(acce_X.head(), '\n', 'y=', acce_y)


        gyro_X = gyro_res[['gyro_x', 'gyro_y', 'gyro_z']]
        gyro_y = gyro_res[['activity']] 
        gyro_y = np.ravel(gyro_y)     # change to array
        print(gyro_X.head(), '\n', 'y=', gyro_y)


        # test dataset
        '''
        path2 = 'D:/WORK4/watch_predict-1.0/input_data/0_Accelerometer_Sat-Apr-27-15_24_38-GMT08_00-2019_Log.csv'
        res_test = Read_from_csv(path2)
        X2 = res_test[['acceleration_x', 'acceleration_y', 'acceleration_z']]
        print(X2.head())
        '''

        acce_X_train, acce_X_test, acce_y_train, acce_y_test = train_test_split(acce_X, acce_y, test_size=0.2, random_state=0)
        gyro_X_train, gyro_X_test, gyro_y_train, gyro_y_test = train_test_split(gyro_X, gyro_y, test_size=0.2, random_state=0)

        # test data(Decision Tree / Random forest)
        model_For_acce = Set_Random_Forest(acce_X_train, acce_X_test, acce_y_train, acce_y_test)
        model_For_gyro = Set_Random_Forest(gyro_X_train, gyro_X_test, gyro_y_train, gyro_y_test)

    except Exception as e:

        print('exception type is:')
        print(type(e), str(e))
        print('exception happened')


    joblib.dump(model_For_acce, 'saved_model/md1_acce.pkl')
    joblib.dump(model_For_gyro, 'saved_model/md2_gyro.pkl')
    print('dump finish!')


if __name__ == "__main__":
    main()