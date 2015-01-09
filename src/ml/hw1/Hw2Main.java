/**
 * 
 */
package ml.hw1;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 29, 2014 [Sousa]
 */
public class Hw2Main {

//	protected static final String DATA_INPUT_DIR = "src/main/resources/MlHw1Data/test_data_20newsgroups.csv";
//	protected static final String LABELS_INPUT_DIR = "src/main/resources/MlHw1Data/test_labels_20newsgroups.csv";
//	protected static final int[] news20ClassLabels = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	
//	protected static final String DATA_INPUT_DIR = "src/main/resources/MlHw1Data/data_20newsgroups.csv";
//	protected static final String LABELS_INPUT_DIR = "src/main/resources/MlHw1Data/labels_20newsgroups.csv";
//	protected static final int NROFSPLITS = 10;
//	protected static final String TRAIN_PERCENT = "5 10 15 20 25 30";
	protected static RealMatrix[] usefulData;
	protected static RealMatrix yMatrix;
	protected static int[] news20ClassLabels;
	protected static int nrOfSplits;
	
	protected static String[] wordsInDoc;
	
	public static void main(String[] args) throws IOException {

		//Initiate data parameters
		
		RealMatrix dataMatrix = usefulData[0];	
		int nrOfTrainPercents = args.length -4;
		
		double[] train_Percent = new double[nrOfTrainPercents];
		for(int i=0; i< nrOfTrainPercents; i++)	{
			train_Percent[i] = Double.parseDouble(args[i+4]);
		}
		
		if(args[0].equals("0"))	{
			//Run Logistic regression
			LogisticRegressionClass.runMultiClass(dataMatrix,yMatrix,nrOfSplits,train_Percent);			
		}
		else	{
			//Run Naive Bayes
			RealMatrix classToWordMatrix = usefulData[1];
			NaiveBayesClassifier.runMultiClass(dataMatrix, yMatrix, classToWordMatrix, news20ClassLabels, nrOfSplits, train_Percent);
			
		}
		

		System.out.println("Test Complete.");
	}

	/**
	 * @param string
	 * @throws IOException 
	 */
	private static void preprocess20NewsGroupsData(String dataFilePath, String outputDirectory) throws IOException {

		List<String> inputLines = FileUtils.readLines(new File(dataFilePath));
		String outputString = "";
		int currentDocId = 0; 
		RealMatrix preprocessedDatamatrix = new Array2DRowRealMatrix();
		
		for(String inputLine : inputLines)	{
			String[] splitLine = inputLine.split(",");
			if(Integer.parseInt(splitLine[0]) == currentDocId)	{
				//Add to current line
				outputString += splitLine[1];
			}
			else	{
				//Start new line
			}
		}
		
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDirectory),"UTF-8"));
		writer.write(outputString);	
		writer.close();	
		
	}

}
