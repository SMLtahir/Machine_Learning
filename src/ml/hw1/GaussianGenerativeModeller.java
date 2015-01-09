/**
 * 
 */
package ml.hw1;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
//import org.hibernate.cfg.ClassPropertyHolder;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 27, 2014 [Sousa]
 */
public class GaussianGenerativeModeller {

	protected static double[] getGaussianAkValues(RealMatrix withinClassCovariance, 
			RealMatrix[] orderedClassMeans, RealMatrix x_matrix, double[] classPriorProbs)	{
		
		int nrOfClasses = orderedClassMeans.length;
		RealMatrix[] gaussianWkMatrices = new RealMatrix[nrOfClasses];
		double[] gaussianWk0Values = new double[nrOfClasses];
		double[] gaussianAkValues = new double[nrOfClasses];
		
		RealMatrix withinClassCovarianceInverse = MatrixUtils.inverse(withinClassCovariance);
		
		for(int k=0; k< nrOfClasses; k++)	{
		
			double logOfPriorProb = Math.log(classPriorProbs[k]);
			RealMatrix kthMeanTranspose = orderedClassMeans[k].transpose();			
			RealMatrix gaussianWk0MatrixMain = kthMeanTranspose.multiply(withinClassCovarianceInverse)
					.multiply(orderedClassMeans[k])
					.scalarMultiply(-0.5);
			
			gaussianWkMatrices[k] = withinClassCovarianceInverse.multiply(orderedClassMeans[k]);
			gaussianWk0Values[k] = gaussianWk0MatrixMain.getEntry(0, 0) + logOfPriorProb;
			
			gaussianAkValues[k] = gaussianWkMatrices[k].transpose()
					.multiply(x_matrix)
					.getEntry(0, 0) 
					+ gaussianWk0Values[k]; 
		}
		
		return gaussianAkValues;
	}
	
	/*protected static double[] getPosteriorClassProbs(RealMatrix[] orderedClassMeans, 
			RealMatrix withinClassCovariance, RealMatrix x_matrix, double[] classPriorProbs)	{
		
		int nrOfClasses = classPriorProbs.length;
		
		double[] akValues = getGaussianAkValues(withinClassCovariance, orderedClassMeans, x_matrix, classPriorProbs);
		double[] posteriorClassProbs = new double[nrOfClasses];
		
		for(int k=0; k< nrOfClasses; k++)	{
			double sumOfAkValues = 0.0;	
			for(int j=0; j< nrOfClasses; j++)	{
				sumOfAkValues += Math.exp(akValues[j]- akValues[k]);
			}	
			posteriorClassProbs[k] = 1.0/sumOfAkValues;
		}
		
		return posteriorClassProbs;
	}*/
	
	protected static double getGaussianPredictedClassLabel(double[] posteriorClassProbs, double[] orderedClassLabels)	{
		
		int nrOfClasses = orderedClassLabels.length;
		int highestProbClass = 0;
		
		for(int k=1; k< nrOfClasses; k++)	{
			if(posteriorClassProbs[k] > posteriorClassProbs[highestProbClass])
				highestProbClass = k;
		}
		return orderedClassLabels[highestProbClass];
	}
	
	protected static double[] getVariance(RealMatrix xHatMatrix, RealMatrix yMatrix, RealMatrix[] orderedClassMeans, int[] orderedClassInstances)	{
		int nrOfClasses = orderedClassInstances.length;
		double[] sumOfMeanSquareDiff = getSumOfMeanSquareDiff(xHatMatrix, yMatrix, orderedClassMeans, orderedClassInstances);
		double[] variance = new double[nrOfClasses];
		
		for(int k=0; k< nrOfClasses; k++)	{
			variance[k] = Math.sqrt(sumOfMeanSquareDiff[k]/ (double) orderedClassInstances[k]);
		}
		
		return variance;
	}
	
	protected static double[] getSumOfMeanSquareDiff(RealMatrix xHatMatrix, RealMatrix yMatrix, RealMatrix[] orderedClassMeans, int[] orderedClassInstances)	{
		
		int totalNrOfInstances = yMatrix.getRowDimension();
		int nrOfClasses = orderedClassInstances.length;
		double[] sumOfMeanSquareDiff = new double[nrOfClasses];
		
		for(int i=0; i< totalNrOfInstances; i++)	{
			for(int k=0; k< nrOfClasses; k++)	{
				if(Math.round(yMatrix.getEntry(i,0)) == k)	{
					sumOfMeanSquareDiff[k] += Math.pow((xHatMatrix.getEntry(i,0) - orderedClassMeans[k].getEntry(0,0)), 2);
				}
			}
		}
		
		return sumOfMeanSquareDiff;
	}

	/**
	 * @return
	 */
	protected static double[] get2ClassGaussianConditionalProbs(RealMatrix xHatMatrix, RealMatrix yMatrix, RealMatrix[] orderedClassMeans, int[] orderedClassInstances) {

		double[] sumOfMeanSquareDiff = getSumOfMeanSquareDiff(xHatMatrix, yMatrix, orderedClassMeans, orderedClassInstances);
		double[] variance = getVariance(xHatMatrix, yMatrix, orderedClassMeans, orderedClassInstances);
		int nrOfClasses = orderedClassInstances.length;
		
		double[] classConditionalProbs = new double[nrOfClasses];
		
		for(int k=0; k< nrOfClasses; k++)	{
			classConditionalProbs[k] = 1.0/Math.sqrt(2*3.14*variance[k])*Math.exp(-sumOfMeanSquareDiff[k]/(2*variance[k]));
		}
		
		return classConditionalProbs;
	}

	/**
	 * @param orderedClassInstances
	 * @param totalNrOfInstances
	 * @return
	 */
	protected static double[] getClassPriorProbs(int[] orderedClassInstances,
			int totalNrOfInstances) {

		int nrOfClasses = orderedClassInstances.length;
		double[] classPriorProbs = new double[nrOfClasses];
		
		for(int k=0; k< nrOfClasses; k++)	{
			classPriorProbs[k] = (double) orderedClassInstances[k]/(double) totalNrOfInstances;
		}
		
		return classPriorProbs;
	}

	/**
	 * @param classPriorProbs
	 * @param classConditionalProbs
	 * @return
	 */
	protected static double[] getClassPosteriorProbs(double[] classPriorProbs,
			double[] classConditionalProbs) {

		int nrOfClasses = classConditionalProbs.length;
		double[] classPosteriorProbs = new double[nrOfClasses];
		double probabilitySum = 0.0;
		
		for(int k=0; k< nrOfClasses; k++)	{
			probabilitySum += classConditionalProbs[k]*classPriorProbs[k];
		}
		
		for(int k=0; k< nrOfClasses; k++)	{
			classPosteriorProbs[k] = classConditionalProbs[k]*classPriorProbs[k]/probabilitySum;
		}
		return classPosteriorProbs;
	}

	/**
	 * @param xHat
	 * @param xHatOrderedMeans
	 * @param xHatCovariance
	 * @param length 
	 * @return
	 */
	protected static double[] get4ClassGaussianConditionalProbs(RealMatrix xHat,
			RealMatrix[] xHatOrderedMeans, RealMatrix xHatCovariance, int nrOfClasses) {

		double[] classConditionalProbs = new double[nrOfClasses];
		int nrOfDimensions = xHatOrderedMeans.length;
		
		SingularValueDecomposition xHatCovSvd = new SingularValueDecomposition(xHatCovariance);
		EigenDecomposition xHatEvd = new EigenDecomposition(xHatCovariance);
		double covarDet = xHatEvd.getDeterminant();
		
		RealMatrix xHatMeanIthCol =  new Array2DRowRealMatrix(xHat.getRowDimension(),1);
		for(int k=0; k< nrOfClasses; k++)	{
			RealMatrix xHatMeanColumnMatrix = new Array2DRowRealMatrix(xHat.getRowDimension(),xHat.getColumnDimension());
			
			RealMatrix xHatMean0thCol = xHatMeanIthCol.scalarAdd(xHatOrderedMeans[k].getEntry(0,0));
			RealMatrix xHatMean1stCol = xHatMeanIthCol.scalarAdd(xHatOrderedMeans[k].getEntry(1,0));
			RealMatrix xHatMean2ndCol = xHatMeanIthCol.scalarAdd(xHatOrderedMeans[k].getEntry(2,0));
			
			xHatMeanColumnMatrix.setColumnMatrix(0, xHatMean0thCol);
			xHatMeanColumnMatrix.setColumnMatrix(1, xHatMean1stCol);
			xHatMeanColumnMatrix.setColumnMatrix(2, xHatMean2ndCol);
			
			double constantTerm = Math.sqrt(covarDet)/Math.pow(2*3.14,nrOfDimensions/2);
			RealMatrix meanDifference = xHat.subtract(xHatMeanColumnMatrix);
			double expTerm = meanDifference
					.multiply(xHatCovSvd.getSolver().getInverse())
					.multiply(meanDifference.transpose())
					.scalarMultiply(-0.5)
					.getEntry(0, 0);
			
			classConditionalProbs[k] = constantTerm*expTerm;
		}
		
		return classConditionalProbs;
	}
}
