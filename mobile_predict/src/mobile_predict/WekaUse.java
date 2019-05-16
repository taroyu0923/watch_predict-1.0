package mobile_predict;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;



public class WekaUse {

	public static double getResult(Instance inputNum, J48 model) throws FileNotFoundException, Exception 
	{
		Instance dataRaw = inputNum;
		J48 rf = model;
		double label = rf.classifyInstance(dataRaw);
        
        System.out.println(label);
        
		
		return label;
	}
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String path = "Accle_model_build1.model";
		double resultNum;
		J48 rf = (J48) SerializationHelper.read(new FileInputStream(path));
		
		
		/*
		 * input hint!
		 * 1. Announce a ArrayList<Attribute> object
		 * 2. Add your attribute
		 * 3. Announce a empty Instances
		 * 4. Set your attributes
		 * 5. Create a DenseInstance & add it into your empty Instances
		 * 6. Call function
		 * */
		//Announce ArrayList<Attribute>
		ArrayList<Attribute> adjust = new ArrayList<Attribute>();
		adjust.add(new Attribute("accr_x"));
		adjust.add(new Attribute("accr_y"));
		adjust.add(new Attribute("accr_z"));
		
		//Announce instances
		Instances data = new Instances("TestInstances", adjust, 0);
		data.setClassIndex(data.numAttributes() - 1);

		//Set attributes
		double testValue[] = {-10.27, -6.35, -0.225};
		
		//Create a DenseInstance
		Instance testInstance = new DenseInstance(1.0, testValue);
		testInstance.setDataset(data);
		
		//Call function
		resultNum = getResult(testInstance, rf);
		

	}

}
