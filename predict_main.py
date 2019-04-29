from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import csv

def Read_from_csv(path):
    test_array = pd.read_csv(path)
    print(test_array.head)
    return test_array


def Read_from_pickle(path):
	model = joblib.load(path)
	return model


def Predict_result(model, data):
	result = model.predict(data)
	return result


# main
def main():

	'''
	
	'''


	test_path = 'D:/WORK4/watch_predict-1.0/input_data/test_1.csv'		# input your test data
	test_model_path = 'D:/WORK4/watch_predict-1.0/saved_model/md2_Ram_For.pkl'		# input your model

	test_data_temp = Read_from_csv(test_path)
	test_data = test_data_temp[['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']]	# input your attribute

	
	test_model = Read_from_pickle(test_model_path)
	test_result = Predict_result(test_model, test_data)
	test_result = [test_result]
	print(test_result)

	'''
	test_num = [[0.2, 0.4, 0.6, 1.2, 0.8, 0.2]] # test for double array
	test_result_1 = Predict_result(test_model, test_num)
	print(test_result_1)
	'''


	with open('output_result.csv', 'w', newline='') as csvfile:

		writer = csv.writer(csvfile)
		writer.writerow(['result'])
		writer.writerows(test_result)



if __name__ == "__main__":
    main()