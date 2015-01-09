/**
 * 
 */
package ml.hw1;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 29, 2014 [Sousa]
 */
public class LinearDiscriminant {

	protected static RealMatrix transformYMatrix(RealMatrix y_matrix, double[] classLabels)	{
		
		int totalNrOfInstances = y_matrix.getRowDimension();
		RealMatrix transformedYMatrix = new Array2DRowRealMatrix(totalNrOfInstances, classLabels.length);
		Map<Double, Integer> columnIndex = new HashMap<Double, Integer>();
		
		for(int k=0; k< classLabels.length; k++)	{
			columnIndex.put(classLabels[k], k);	
		}
		
		for(int i=0; i< totalNrOfInstances; i++)	{
			transformedYMatrix.setEntry(i, columnIndex.get(y_matrix.getEntry(i, 0)), 1);
		}
		return transformedYMatrix;
	}

	/**
	 * @param predictedYMatrix
	 * @param testYMatrix
	 * @return
	 */
	protected static double[] getPredictions(RealMatrix predictedYMatrix,
			RealMatrix testYMatrix) {

		int totalNrOfInstances = predictedYMatrix.getRowDimension();
		int nrOfAttributes = predictedYMatrix.getColumnDimension();
		double[] predictions = new double[totalNrOfInstances];
		
		for(int i=0; i< totalNrOfInstances; i++)	{
			double bestRankedPrediction = -999.00;
			int predictionIndex = 0;
			
			for(int k=0; k< nrOfAttributes; k++)	{
				if(predictedYMatrix.getEntry(i, k) > bestRankedPrediction)	{
					bestRankedPrediction = predictedYMatrix.getEntry(i, k);
					predictionIndex = k;
				}
			}
			predictions[i] = (double) predictionIndex;
		}
		
		return predictions;
	}
	
}
