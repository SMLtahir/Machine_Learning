package ml.hw3;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

public class myRForest2 {

//	private static final String FILENAME = "src/ml/Hw3/Mushroom.csv";
//	private static final String FILENAME = "src/ml/Hw3/testData.csv";
	private static String FILENAME;
	private static final int B = 100;
	protected static RealMatrix yMatrix;
	protected static RealMatrix xMatrix;
	private static final int NR_OF_FOLDS = 10;
	private static final int NR_OF_LEVELS = 2;					//Nr of levels in Decision Tree
	private static final int[] M = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22};
//	private static final int[] M = {2};
	
	public static void main(String[] args) throws IOException {

		int argsLength = args.length;
		FILENAME = args[0];
		int[] M = new int[argsLength-1];
		for(int i=1;i< argsLength; i++)	{
			M[i-1] = Integer.parseInt(args[i]);
		}
		int nrOfBaseClassifiers = B;
		
		//Initialize data instances
		List<String> instances = new InputOutput().initInstances(FILENAME);
		yMatrix = HwMain.yMatrix;
		xMatrix = HwMain.xMatrix;
		int totalNrOfInstances = yMatrix.getRowDimension();
		int totalNrOfAttributes = xMatrix.getColumnDimension();
		

		double avgAllMTrainError = 0.0;
		double avgAllMTestError = 0.0;
		double[] trainAverageErrorPerM = new double[argsLength - 1];
		double[] testAverageErrorPerM = new double[argsLength - 1];
		//Do the runs for different number of base classifiers
		for(int k=0; k< M.length; k++)	{
			
			double[] trainFoldErrors = new double[NR_OF_FOLDS];
			double[] testFoldErrors = new double[NR_OF_FOLDS];

			System.out.println("Starting run for M = "+ M[k]);
			int nrOfUsableIds = M[k];
			int nrOfRestrictedIds = totalNrOfAttributes - nrOfUsableIds;
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
				RealMatrix[] baseClassifierParams = new Array2DRowRealMatrix[nrOfBaseClassifiers];
				ArrayList<Integer> restrictedAttriIds = new ArrayList<Integer>();
				
//				System.out.println("Building base classifiers...");
				for(int baseClsfrIndex=0; baseClsfrIndex< nrOfBaseClassifiers; baseClsfrIndex++)	{

					RealMatrix[] sampledXYMatrix = Hw3Utils.getBootstrappedSample(nrOfTrainInstances, trainXMatrix, trainYMatrix);
					
					restrictedAttriIds = Hw3Utils.getRestrictedIds(totalNrOfAttributes, nrOfRestrictedIds);
					if(M[k]> 1 && M[k]<= totalNrOfAttributes)
						baseClassifierParams[baseClsfrIndex] = DecisionTree.train2DecisionTree(baseClsfrIndex, sampledXYMatrix, restrictedAttriIds, NR_OF_LEVELS);
					else if(M[k]== 1)
						baseClassifierParams[baseClsfrIndex] = DecisionTree.train1DecisionTree(baseClsfrIndex, sampledXYMatrix, restrictedAttriIds, NR_OF_LEVELS);
					else	{
						System.out.println("Please input correct value of M - between 1 and number of attributes.");
						break;
					}
						
				}	
				
				////////////////////////////////////////////////////////////////////////////////////////
				//Predict train instances
				int correctTrainPredictions = 0;
				int wrongTrainPredictions = 0;
				for(int i=0; i< trainXMatrix.getRowDimension(); i++)	{
					int predictedLabel = Hw3Utils.getBaggedPrediction(baseClassifierParams, nrOfBaseClassifiers, trainXMatrix, i);
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
				trainAverageErrorPerM[k] += trainFoldErrors[fold];
				
				////////////////////////////////////////////////////////////////////////////////////////
				
				//Predict test instances
				int correctTestPredictions = 0;
				int wrongTestPredictions = 0;
				for(int i=0; i< testXMatrix.getRowDimension(); i++)	{
					int predictedLabel = Hw3Utils.getBaggedPrediction(baseClassifierParams, nrOfBaseClassifiers, testXMatrix, i);
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
				testAverageErrorPerM[k] += testFoldErrors[fold];
				////////////////////////////////////////////////////////////////////////////////////////
			}
			StandardDeviation stdDev = new StandardDeviation();
			double trainErrorStdDev = stdDev.evaluate(trainFoldErrors);
			double averageTrainErrorAcrossFolds = trainAverageErrorPerM[k]/ (double)NR_OF_FOLDS;
			avgAllMTrainError += averageTrainErrorAcrossFolds; 
			System.out.println("\nFor M = "+ M[k]+ " Train Average error percentage across all 10 folds: "+ averageTrainErrorAcrossFolds+ " % with standard deviation: "+ trainErrorStdDev);
			
			double testErrorStdDev = stdDev.evaluate(testFoldErrors);
			double averageTestErrorAcrossFolds = testAverageErrorPerM[k]/ (double)NR_OF_FOLDS;
			avgAllMTestError += averageTestErrorAcrossFolds;
			System.out.println("For M = "+ M[k]+ " Average test error percentage across all 10 folds: "+ averageTestErrorAcrossFolds+ " % with standard deviation: "+ testErrorStdDev);
			System.out.println("\n");
		}
		StandardDeviation stdDev = new StandardDeviation();
		double trainAllMErrorStdDev = stdDev.evaluate(trainAverageErrorPerM);
		double testAllMErrorStdDev = stdDev.evaluate(testAverageErrorPerM);
		
		System.out.println("Average train error % across all values of M: "+ avgAllMTrainError/(double)(argsLength-1)+ " with standard deviation: "+ trainAllMErrorStdDev/ (double)NR_OF_FOLDS);
		System.out.println("Average test error % across all values of M: "+ avgAllMTestError/(double)(argsLength-1)+ " with standard deviation: "+ testAllMErrorStdDev/ (double)NR_OF_FOLDS);
		System.out.println("End of all runs.");

	

	}

}
