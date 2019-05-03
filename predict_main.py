from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import pandas as pd
import numpy as np



'''


Parameters
--------


Read_from_csv: string, optional(file's path, only for .csv file)
    Read the csv file and put them into dataframe

Read_from_pickle: string, optional(file's path, only for .pkl file)
    Read the model file and put them into process

Predict_result: model(pkl), data(Double array)
	use input data and model to predict result and output

output_csv: data(array), string(path)
	write result data into the csv file, then output


'''



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


def output_csv(data_predict, file_path):
	data_adjust = pd.DataFrame(data_predict)
	data_adjust.to_csv(file_path, index=False)


# main
def main():

	'''

	Main function
	--------

	Set the path of input & output file:

	1. input data (csv file)
	2. input model (pickle file)
	3. input result file's path
	4. choose the attribute that you want to predict
	5. predict data
	6. output result


	'''
	# input data
	acce_test_path = 'D:/Year_PROJ/watch_predict-1.0/input_data/Accelerometer_Merge_Log.csv'		# input your test data path
	gyro_test_path = 'D:/Year_PROJ/watch_predict-1.0/input_data/Gyro_Merge_Log.csv'		# input your test data path

	# input model
	acce_model_path = 'D:/Year_PROJ/watch_predict-1.0/saved_model/md1_acce.pkl'		# input your model path
	gyro_model_path = 'D:/Year_PROJ/watch_predict-1.0/saved_model/md2_gyro.pkl'		# input your model path

	# output data
	acce_output_path = 'output_data/acce_result.csv'
	gyro_output_path = 'output_data/gyro_result.csv'

	# read file & choose attribute
	acce_temp = Read_from_csv(acce_test_path)
	gyro_temp = Read_from_csv(gyro_test_path)
	acce_data = acce_temp[['acceleration_x', 'acceleration_y', 'acceleration_z']]	# input attribute you want to predict(acceleration data)
	gyro_data = gyro_temp[['gyro_x', 'gyro_y', 'gyro_z']]	# input attribute you want to predict(gyro data)
	acce_model = Read_from_pickle(acce_model_path)
	gyro_model = Read_from_pickle(gyro_model_path)

	# predict
	acce_result = Predict_result(acce_model, acce_data)
	gyro_result = Predict_result(gyro_model, gyro_data)
	print(acce_result, '\n', gyro_result)

	# output
	output_csv(acce_result, acce_output_path)
	output_csv(gyro_result, gyro_output_path)



if __name__ == "__main__":
    main()