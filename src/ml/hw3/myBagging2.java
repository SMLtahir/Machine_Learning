package ml.hw3;

import java.awt.PageAttributes.OriginType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import javax.swing.tree.TreeNode;

import ml.hw3.TrainTestSetup;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

public class myBagging2 {

//	private static final String FILENAME = "src/ml/Hw3/Mushroom.csv";
//	private static final String FILENAME = "src/ml/Hw3/testData.csv";
//	private static final int[] B = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
//	private static final int[] B = {5};
	private static String FILENAME;
	protected static RealMatrix yMatrix;
	protected static RealMatrix xMatrix;
	private static final int NR_OF_FOLDS = 10;
	private static final int NR_OF_LEVELS = 2;					//Nr of levels in Decision Tree
	
	public static void main(String[] args) throws IOException {

		int argsLength = args.length;
		FILENAME = args[0];
		int nrOfBaseClassifiers = argsLength- 1;
		int[] B = new int[nrOfBaseClassifiers];
		for(int i=1;i< argsLength; i++)	{
			B[i-1] = Integer.parseInt(args[i]);
		}
		
		
		//Initialize data instances
		List<String> instances = new InputOutput().initInstances(FILENAME);
		yMatrix = HwMain.yMatrix;
		xMatrix = HwMain.xMatrix;
		int totalNrOfInstances = yMatrix.getRowDimension();
		int totalNrOfAttributes = xMatrix.getColumnDimension();


		double avgAllBTrainError = 0.0;
		double avgAllBTestError = 0.0;
		double[] trainAverageErrorPerB = new double[argsLength - 1];
		double[] testAverageErrorPerB = new double[argsLength - 1];
		
		//Do the runs for different number of base classifiers
		for(int k=0; k< B.length; k++)	{

			System.out.println("Starting run for B = "+ B[k]);
			double[] trainFoldErrors = new double[NR_OF_FOLDS];
			double[] testFoldErrors = new double[NR_OF_FOLDS];
			
			//Run fold iterations
			for(int fold=0; fold< NR_OF_FOLDS; fold++)	{
				
				//Generate test sets
				String[] testSets = TrainTestSetup.createNFoldCVSetup(totalNrOfInstances, NR_OF_FOLDS);
				List<Integer> testInstanceIds = TrainTestSetup.getTestInstanceIds(testSets, fold);
				int nrOfTestInstances = testInstanceIds.size();
				int nrOfTrainInstances = totalNrOfInstances - nrOfTestInstances;
				
				RealMatrix[] trainTestDataMatrices = TrainTestSetup.getTrainTestMatrices(xMatrix, yMatrix, testInstanceIds);
				RealMatrix trainXMatrix = trainTestDataMatrices[0];
				RealMatrix trainYMatrix = trainTestDataMatrices[1];
				RealMatrix testXMatrix = trainTestDataMatrices[2];
				RealMatrix testYMatrix = trainTestDataMatrices[3];
			
				//Nr of nodes = 2^0 + 2^1 = 3 for a 2-level decision tree
				//Format of classifierParams: [NrOfBaseClassifiers][NrOfNodes][2 cells to store ColumnOfDiscriminatingAttribute and SplitPointValue]
//				double[][] classifierParams = new double[ B[k] ][3];
				RealMatrix[] baseClassifierParams = new Array2DRowRealMatrix[ B[k] ];
				ArrayList<Integer> restrictedAttriIds = new ArrayList<Integer>();
				
				for(int i=0; i< B[k]; i++)	{
					//Size of sample will be = totalNrOfInstances
					RealMatrix[] sampledXYMatrix = Hw3Utils.getBootstrappedSample(nrOfTrainInstances, trainXMatrix, trainYMatrix);
					restrictedAttriIds = Hw3Utils.getRestrictedIds(totalNrOfAttributes, 0);			//Select all attributes		   
					baseClassifierParams[i] = DecisionTree.train2DecisionTree(i, sampledXYMatrix, restrictedAttriIds,NR_OF_LEVELS);
				}	
				
				////////////////////////////////////////////////////////////////////////////////////////
				//Predict train instances
				int correctTrainPredictions = 0;
				int wrongTrainPredictions = 0;
				for(int i=0; i< trainXMatrix.getRowDimension(); i++)	{
					int predictedLabel = Hw3Utils.getBaggedPrediction(baseClassifierParams, B[k], trainXMatrix, i);
					int actualLabel = (int) trainYMatrix.getEntry(i, 0);
					
//					System.out.println("Pred. Label: "+ predictedLabel+ " Actual Label: "+ actualLabel);					
//					System.out.print("");
					if(predictedLabel == actualLabel)
						correctTrainPredictions++;
					else
						wrongTrainPredictions++;
				}
				
				double trainAccuracy = (double) 100*correctTrainPredictions/ (double) (correctTrainPredictions+wrongTrainPredictions);
//				System.out.println("Fold "+ (fold+1)+ " complete");
				trainFoldErrors[fold] = 100-trainAccuracy;
				System.out.println("Train Error % in Fold: "+ (fold+1)+ " is: "+ trainFoldErrors[fold]);
				trainAverageErrorPerB[k] += trainFoldErrors[fold];
				
				////////////////////////////////////////////////////////////////////////////////////////
				
				//Predict test instances
				int correctTestPredictions = 0;
				int wrongTestPredictions = 0;
				for(int i=0; i< testXMatrix.getRowDimension(); i++)	{
					int predictedLabel = Hw3Utils.getBaggedPrediction(baseClassifierParams, B[k], testXMatrix, i);
					int actualLabel = (int) testYMatrix.getEntry(i, 0);
					
//					System.out.println("Pred. Label: "+ predictedLabel+ " Actual Label: "+ actualLabel);					
//					System.out.print("");
					if(predictedLabel == actualLabel)
						correctTestPredictions++;
					else
						wrongTestPredictions++;
				}
				
				double testAccuracy = (double) 100*correctTestPredictions/ (double) (correctTestPredictions+wrongTestPredictions);
//				System.out.println("Fold "+ (fold+1)+ " complete");
				testFoldErrors[fold] = 100-testAccuracy;
				System.out.println("Test Error % in Fold: "+ (fold+1)+ " is: "+ testFoldErrors[fold]);
				testAverageErrorPerB[k] += testFoldErrors[fold];
				
				////////////////////////////////////////////////////////////////////////////////////////				
			}
			StandardDeviation stdDev = new StandardDeviation();
			double trainErrorStdDev = stdDev.evaluate(trainFoldErrors);
			double averageTrainErrorAcrossFolds = trainAverageErrorPerB[k]/ (double)NR_OF_FOLDS;
			avgAllBTrainError += averageTrainErrorAcrossFolds; 
			System.out.println("\nFor B = "+ B[k]+ " Train Average error percentage across all 10 folds: "+ averageTrainErrorAcrossFolds+ " % with standard deviation: "+ trainErrorStdDev);
			
			double testErrorStdDev = stdDev.evaluate(testFoldErrors);
			double averageTestErrorAcrossFolds = testAverageErrorPerB[k]/ (double)NR_OF_FOLDS;
			avgAllBTestError += averageTestErrorAcrossFolds;
			System.out.println("For B = "+ B[k]+ " Test Average error percentage across all 10 folds: "+ averageTestErrorAcrossFolds+ " % with standard deviation: "+ testErrorStdDev);
			System.out.println("\n");
		}
		StandardDeviation stdDev = new StandardDeviation();
		double trainAllBErrorStdDev = stdDev.evaluate(trainAverageErrorPerB);
		double testAllBErrorStdDev = stdDev.evaluate(testAverageErrorPerB);
		
		System.out.println("Average train error % across all values of B: "+ avgAllBTrainError/(double)(argsLength-1)+ " with standard deviation: "+ trainAllBErrorStdDev/ (double)NR_OF_FOLDS);
		System.out.println("Average test error % across all values of B: "+ avgAllBTestError/(double)(argsLength-1)+ " with standard deviation: "+ testAllBErrorStdDev/ (double)NR_OF_FOLDS);
		System.out.println("End of all runs.");

	}

	

	

	

}
