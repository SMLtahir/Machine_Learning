/**
 * 
 */
package ml.hw1;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 29, 2014 [Sousa]
 */
public class Hw1_3_2_2Class {

//	protected static final String SPAM_INPUT_DIR = "src/main/resources/MlHw1Data/spam.csv";
//	protected static final int NR_OF_FOLDS = 10;
	protected static final double[] classLabels = Hw1Main.givenClassLabels;

	public static void main(String[] args) throws IOException {

		//Initialize data instances
		int NR_OF_FOLDS = Integer.parseInt(args[1]);
		
		int totalNrOfInstances = Hw1Main.yMatrix.getRowDimension();
		int nrOfAttributes = Hw1Main.xMatrix.getColumnDimension();
		
		//Generate test sets
		String[] testSets = TrainTestSetup.createNFoldCVSetup(totalNrOfInstances, NR_OF_FOLDS);
		
		double foldAverageError = 0.0;
		double[] foldErrors = new double[NR_OF_FOLDS];
		
		for(int fold=0; fold< NR_OF_FOLDS; fold++)	{
			List<Integer> testInstanceIds = TrainTestSetup.getTestInstanceIds(testSets, fold);
			int nrOfTestInstances = testInstanceIds.size();
			int nrOfTrainInstances = totalNrOfInstances - nrOfTestInstances;
			
			RealMatrix[] trainTestDataMatrices = TrainTestSetup.getTrainTestMatrices(Hw1Main.xMatrix, Hw1Main.yMatrix, testInstanceIds);
			
			RealMatrix trainXMatrix = trainTestDataMatrices[0];
			RealMatrix trainYMatrix = trainTestDataMatrices[1];
			RealMatrix testXMatrix = trainTestDataMatrices[2];
			RealMatrix testYMatrix = trainTestDataMatrices[3];
			
			//Train the model
			int[] orderedTrainClassInstances = MatrixOperators.getOrderedClassInstances(
					trainXMatrix, trainYMatrix, classLabels, nrOfTrainInstances);
			RealMatrix[] orderedTrainClassMeans = MatrixOperators.getOrderedClassMeans(
					trainXMatrix, trainYMatrix, classLabels, nrOfTrainInstances, orderedTrainClassInstances);
			RealMatrix withinClassCovariance = FisherDiscriminant.getWithinClassCovariance(
					trainXMatrix, trainYMatrix, orderedTrainClassMeans, nrOfTrainInstances, nrOfAttributes, classLabels);
			
			RealMatrix withinClassCovarianceInverse = MatrixUtils.inverse(withinClassCovariance);
			RealMatrix meanDifference = orderedTrainClassMeans[1].subtract(orderedTrainClassMeans[0]);
			
			RealMatrix fishersW = withinClassCovarianceInverse.multiply(meanDifference);
			
			//Test the model
			RealMatrix xHat = testXMatrix.multiply(fishersW);
			int[] orderedTestClassInstances = MatrixOperators.getOrderedClassInstances(
					xHat, testYMatrix, classLabels, nrOfTestInstances);
			
			RealMatrix[] xHatOrderedMeans = MatrixOperators.getOrderedClassMeans(
					xHat, testYMatrix, classLabels, nrOfTestInstances, orderedTestClassInstances);
			
			double[] classConditionalProbs = GaussianGenerativeModeller.get2ClassGaussianConditionalProbs(
					xHat, testYMatrix, xHatOrderedMeans, orderedTestClassInstances);
			
			double[] classPriorProbs = GaussianGenerativeModeller.getClassPriorProbs(orderedTestClassInstances, nrOfTestInstances);
			
			double[] classPosteriorProbs = GaussianGenerativeModeller.getClassPosteriorProbs(classPriorProbs, classConditionalProbs);
			
			int nrOfCorrectPredictions = 0;
			//Test the model
			for(int i=0; i<nrOfTestInstances; i++)	{
				
				double prediction = GaussianGenerativeModeller.getGaussianPredictedClassLabel(classPosteriorProbs, classLabels);
				
				//System.out.println("Predicted class Label: "+ prediction+ " and Actual: "+ testYMatrix.getEntry(i, 0));
				if(Math.round(prediction) == Math.round(testYMatrix.getEntry(i, 0)))	{
					nrOfCorrectPredictions++;
				}
				
			}
			double accuracy = (double) nrOfCorrectPredictions/(double) nrOfTestInstances;
			foldErrors[fold] = 1-accuracy;
			foldAverageError += foldErrors[fold];
			System.out.println("System accuracy in Fold: "+ (fold+1)+ " is: "+ 100*accuracy+ 
					"% and error rate is: "+ 100*foldErrors[fold]+ "%");
			
		}
		foldAverageError = foldAverageError/NR_OF_FOLDS;
		StandardDeviation stdDev = new StandardDeviation();
		double errorStdDev = stdDev.evaluate(foldErrors);
		System.out.println("Average error Rate over "+ NR_OF_FOLDS+ " folds: "+ 100*foldAverageError+ "%. \nStandard Deviation in error: "+ 100*errorStdDev+ "%");
		
		System.out.println("Test Complete");	

	}

}
