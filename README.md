# watch_predict-1.0


- use Python 3.7
 - package: sklearn, pandas, numpy
- model: md1_Dec_Tree.pkl(決策樹), md2_Ram_For.pkl(隨機森林)
- B_model_original.py 原始的code
- buildmodel_main.py 建立模型的code
- predict_main.py 預測的code


# mobile_predict(use Java)


- use Java SDK 12
- package: java.io, weka.core, weka.classifiers
- model(REPTTree):
 Accle_model_build1.model(走路或跑步模型), Accle_model_machine.model (震動機模型)
    

# input data


-震動機1-7級部分
- 0_ACCELEROMETER_MACH_X.csv, 0_ACCELEROMETER_MACH__POCKETX.csv 是原始的資料(震動機)
- ACCELEROMETER_MACHINE_Merge.csv, ACCELEROMETER_MACHINE_Merge.arff 是前處理後，實際用Python或Weka跑的資料集

-跑步&走路
- 0_ACCELEROMETER_RUN_type-x, 0_ACCELEROMETER_WALK 是原始的資料
- 0_Accelerometer_......_Log.csv, 1_Gyroscope_........_Log.csv 是原始的資料(有手環數據)
- Accelerometer_Merge_Log, Accelerometer_Merge_Log.arff 是前處理後，實際用Python或Weka跑的資料集
- Final_Apr-28-12_15_33_Log.csv 是前處理後，實際用Python或跑的資料集(有手環數據)
- mobile_workflow 是用orange做前處理與測試的資料






