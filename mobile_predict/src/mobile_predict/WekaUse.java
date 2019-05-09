package mobile_predict;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

import weka.core.Attribute;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;



public class WekaUse {

	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String path = "Accle_model.model";
        RandomForest treeClassifier = (RandomForest) SerializationHelper.read(new FileInputStream("Accle_model.model"));
		
		/*DataSource source = new DataSource("Accelerometer_Merge_Log_test.csv");
		Instances testData = source.getDataSet();
		testData.setClassIndex(testData.numAttributes() -1);
		
		
		
		for(int i = 0; i < testData.numInstances(); i++) 
		{
			double testNum = testData.instance(i).classValue();
			String output = testData.classAttribute().value((int) testNum);
			
			Instance newIns = testData.instance(i);
			double pred = treeClassifier.classifyInstance(newIns);
			String predString = testData.classAttribute().value((int) pred);
			System.out.println(predString);
		}*/
		
		
		ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("Accelerometer_Merge_Log_test.csv");
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (test.classIndex() == -1)
            test.setClassIndex(test.numAttributes() - 1);

        
        
        int result = 0;
        
        
        double label = treeClassifier.classifyInstance(test.instance(0));
        
        //test.instance(0).setClassValue(label);
        //System.out.println(test.instance(0).value(3));
        System.out.println(label);
        if(label <= 0.5) 
        {
        	result = 0;
        }
        else 
        {
        	result = 1;
        }
        System.out.println(result);

	}

}
