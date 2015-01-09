/**
 * 
 */
package ml.hw1;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 29, 2014 [Sousa]
 */
public class Hw1_3_3_4Class {

//	protected static final int NR_OF_FOLDS = 10;
	protected static final double[] classLabels = Hw1Main.givenClassLabels;
	
	public static void main(String[] args) throws IOException {

				//Initialize data parameters
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
					RealMatrix transformedYTrainMatrix = LinearDiscriminant.transformYMatrix(trainYMatrix, classLabels);
					RealMatrix transformedYTestMatrix = LinearDiscriminant.transformYMatrix(testYMatrix, classLabels);
					
					RealMatrix trainXMatrixTranspose = trainXMatrix.transpose();
					RealMatrix trainXXTProduct = trainXMatrixTranspose.multiply(trainXMatrix);
					SingularValueDecomposition svd = new SingularValueDecomposition(trainXXTProduct);
					RealMatrix trainXXTProductInverse = svd.getSolver().getInverse();
					RealMatrix trainXTYProduct = trainXMatrixTranspose.multiply(transformedYTrainMatrix);
					RealMatrix leastSquareWStarMatrix = trainXXTProductInverse.multiply(trainXTYProduct);
					
					int nrOfCorrectPredictions = 0;
					
					//Test the model
					RealMatrix predictedYMatrix = testXMatrix.multiply(leastSquareWStarMatrix);
					double[] predictions = LinearDiscriminant.getPredictions(predictedYMatrix, testYMatrix);
					
					for(int i=0; i< nrOfTestInstances; i++)	{
						if(transformedYTestMatrix.getEntry(i, (int) Math.round(predictions[i])) == 1)	{
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
