/**
 * 
 */
package ml.hw1;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 26, 2014 [Sousa]
 */
public class FisherDiscriminant {

	protected static RealMatrix getBetweenClassCovariance(RealMatrix[] orderedClassMeans, int[] orderedClassNrOfInstances,
			RealMatrix totalMean, int nrOfAttributes, int nrOfClasses)	{
		
		RealMatrix betweenClassCovariance = new Array2DRowRealMatrix(nrOfAttributes, nrOfAttributes);
		
		for(int k=0; k< nrOfClasses; k++)	{
			RealMatrix meanDifference = orderedClassMeans[k].subtract(totalMean);
			RealMatrix meanDifferenceTranspose = meanDifference.transpose();
			RealMatrix weightedMeanDifferenceSquare =  new Array2DRowRealMatrix(nrOfAttributes, nrOfAttributes);
			weightedMeanDifferenceSquare = meanDifference.multiply(meanDifferenceTranspose).scalarMultiply(orderedClassNrOfInstances[k]);
			
			betweenClassCovariance = betweenClassCovariance.add(weightedMeanDifferenceSquare);
		}
		
		return betweenClassCovariance;
	}
	
	protected static RealMatrix getWithinClassCovariance(RealMatrix x_matrix, RealMatrix y_matrix, RealMatrix[] orderedClassMeans, 
			int totalNrOfInstances, int nrOfAttributes, double[] orderedClassLabels)	{
		
		RealMatrix withinClassCovariance = new Array2DRowRealMatrix(nrOfAttributes, nrOfAttributes);
		int nrOfClasses = orderedClassLabels.length;
		
		for(int k=0; k< nrOfClasses; k++)	{
			RealMatrix kthClassCovariance = new Array2DRowRealMatrix(nrOfAttributes, nrOfAttributes);
			
			for(int i=0; i< totalNrOfInstances; i++)	{
				if(Math.round(y_matrix.getEntry(i, 0)) == Math.round(orderedClassLabels[k]))	{
					RealMatrix ithVariationFromMean = x_matrix.getRowMatrix(i).transpose().subtract(orderedClassMeans[k]);
					RealMatrix ithVariationFromMeanTranspose = ithVariationFromMean.transpose();
					RealMatrix squaredVariation = ithVariationFromMean.multiply(ithVariationFromMeanTranspose);
					kthClassCovariance = kthClassCovariance.add(squaredVariation);
				}
				else
					continue;
			}
			
			withinClassCovariance = withinClassCovariance.add(kthClassCovariance);
		}
		return withinClassCovariance;
		
	}
	
	protected static RealMatrix get2ClassFishersDiscriminant(RealMatrix withinClassCovariance, 
			RealMatrix firstClassMean, RealMatrix secondClassMean)	{
		
		RealMatrix inverseWithinClassCovariance = MatrixUtils.inverse(withinClassCovariance); 
		RealMatrix fisherMatrix = inverseWithinClassCovariance.multiply(secondClassMean.subtract(firstClassMean));
		return fisherMatrix;
		
	}
	
}
