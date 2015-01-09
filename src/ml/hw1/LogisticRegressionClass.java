/**
 * 
 */
package ml.hw1;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Oct 3, 2014 [Sousa]
 */
public class LogisticRegressionClass {

	/**
	 * @param xMatrix 
	 * @param originalYMatrix 
	 * 
	 */
	protected static double alpha = 0.1;
	protected static int nrOfClasses = Hw2Main.news20ClassLabels.length;
	protected static final double TRAIN_PERCENTAGE = 80;
	
	protected static void runMultiClass(RealMatrix dataMatrix, RealMatrix originalYMatrix, int nrOfSplits, double[] train_Percent) {

		//Initialize data parameters
		int totalNrOfInstances = dataMatrix.getRowDimension();
		int nrOfTrainPercents = train_Percent.length;
		RealMatrix xMatrix = dataMatrix.getSubMatrix(0, dataMatrix.getRowDimension()-1, 1, dataMatrix.getColumnDimension()-1);
		RealMatrix yMatrix = new Array2DRowRealMatrix(totalNrOfInstances, 1);
		
		for(int i=0; i< totalNrOfInstances; i++)	{
			double targetValue = originalYMatrix.getEntry((int) dataMatrix.getEntry(i, 0)-1, 0);
			yMatrix.setEntry(i, 0, targetValue);
		}
		
		double splitAverageErrorRate = 0.0;
		
		//Iterate over the random splits
		for(int split =0; split< nrOfSplits; split++)	{
			System.out.println("Starting split...");
			
			//Set train and test instances based on the train_percent vector
			String testInstancesString = TrainTestSetup.createTrainTestSetup(totalNrOfInstances, TRAIN_PERCENTAGE);
			RealMatrix testYMatrix = TrainTestSetup.getSelectedYMatrix(yMatrix, testInstancesString);
			RealMatrix testXMatrix = TrainTestSetup.getSelectedXMatrix(xMatrix, testInstancesString);
			String[] testInstanceIds = testInstancesString.split(" "); 
			int nrOfTestInstances = testInstanceIds.length;
			double[] splitErrorRate = new double[nrOfTrainPercents];
			double runAverageErrorRate = 0.0;
			
			//Iterate over different train_percent values
			for(int runNr =0; runNr< nrOfTrainPercents; runNr++)	{
				System.out.println("Starting run...");	
				String trainInstances = TrainTestSetup.getTrainInstances(testInstancesString, totalNrOfInstances, train_Percent[runNr]);
				int nrOfTrainInstances = trainInstances.split(" ").length; 
				RealMatrix trainYMatrix = TrainTestSetup.getSelectedYMatrix(yMatrix, trainInstances);
				
				//Initiate the w-Matrix and the gradient matrix
				RealMatrix[] wMatrix = new Array2DRowRealMatrix[20];
				RealMatrix[] gradient = new Array2DRowRealMatrix[nrOfClasses];
				int[] classInstances = new int[20];
				
				for(int k=0; k< nrOfClasses; k++)	{
					wMatrix[k] = new Array2DRowRealMatrix(dataMatrix.getRowDimension(),1);
					gradient[k] = new Array2DRowRealMatrix(dataMatrix.getRowDimension(),1);
				}
				
				//Perform gradient descent
				for(int i=0; i< nrOfTrainInstances; i++)	{
					int classIndex = (int) trainYMatrix.getEntry(i, 0) -1;
					RealMatrix tempMatrix = xMatrix.getRowMatrix(i).transpose()
							.scalarMultiply(trainYMatrix.getEntry(i, 0)
									-Math.signum(xMatrix.multiply(wMatrix[classIndex]).getEntry(0, 0)));
					gradient[classIndex] = gradient[classIndex].add(tempMatrix); 
							
					classInstances[classIndex] ++;
					
				}
				
				//Obtain k W-Matrices and the final optimized w-Matrix
				for(int k=0; k< nrOfClasses; k++)	{
					if(classInstances[k] != 0)
						gradient[k] = gradient[k].scalarMultiply(alpha/(double) classInstances[k]);
					else
						gradient[k] = gradient[k].scalarMultiply(0);
					
					wMatrix[k] = wMatrix[k].subtract(gradient[k]);
				}
				
				//Test the model
				int nrOfPredictions =0;
				for(int i=0; i< nrOfTestInstances; i++)	{
					double[] predictions = new double[nrOfClasses];
					for(int k=0; k< nrOfClasses; k++)	{
						predictions[k] = Math.signum(testXMatrix.multiply(wMatrix[k]).getEntry(0, 0));
					}
					double finalPrediction = getLogRegPrediction(predictions);
					
					if(Math.round(finalPrediction) == testYMatrix.getEntry(i, 0))	{
						nrOfPredictions++;	
					}
				}
				
				double accuracy = 100.0* (double) nrOfPredictions/ (double) nrOfTestInstances;
				double runErrorRate = 100- accuracy;
				runAverageErrorRate += runErrorRate; 
				
				System.out.println("Error Rate: "+ runErrorRate+ " for train_percent= "+ train_Percent[runNr]+ " and split= "+ (split+1));
					
			}
			System.out.println("Error Rate: "+ runAverageErrorRate+ " for the average of all train_percent over split= "+ (split+1));
			splitErrorRate[split] = runAverageErrorRate;
			splitAverageErrorRate += splitErrorRate[split]; 
		}
		System.out.println("Error Rate: "+ splitAverageErrorRate + " for the average of all splits.");
		
		System.out.println("Test complete.");
	}

	/**
	 * @param predictions
	 * @return
	 */
	private static double getLogRegPrediction(double[] predictions) {

		double prediction =-1;
		double tempMax = -1;
		for(int k=0; k< nrOfClasses; k++)	{
			if(predictions[k] >= tempMax)	{
				prediction = k;
				tempMax = predictions[k];
			}
		}
		
		return prediction;
	}

}
