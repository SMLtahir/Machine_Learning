/**
 * 
 */
package ml.hw1;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 26, 2014 [Sousa]
 */
public class MatrixOperators {

	protected static RealMatrix getMeanMatrix(RealMatrix matrix)	{

		int nrOfColumns = matrix.getColumnDimension();
		int nrOfRows = matrix.getRowDimension();
		RealMatrix meanTransposeMatrix = new Array2DRowRealMatrix(1, nrOfColumns);
		RealMatrix meanMatrix = new Array2DRowRealMatrix(1, nrOfColumns);
		
		for(int col=0; col< nrOfColumns; col++)	{
			double valueSum = 0;
			for(int row=0; row< nrOfRows; row++)	{
				valueSum += matrix.getEntry(row, col);
			}	
			double mean = valueSum/(double) nrOfRows;
			meanTransposeMatrix.setEntry(0, col, mean);
		}

		/*System.out.println("Mean Matrix: ");
		InputOutput.printMatrix(meanMatrix);*/
		meanMatrix = meanTransposeMatrix.transpose();
		return meanMatrix;
		
	}
	
	protected static RealMatrix[] getOrderedClassMeans(RealMatrix x_matrix, RealMatrix y_matrix, 
			double[] orderedClassLabels, int totalNrOfInstances, int[] orderedClassInstances)	{
		
		RealMatrix[] orderedClassMeans = new RealMatrix[orderedClassLabels.length];
		
		for(int k=0; k< orderedClassLabels.length; k++)	{
			if(orderedClassInstances[k] == 0)	{
				System.err.println("Error: There are 0 instances of Class "+ orderedClassLabels[k] + " in this fold!");
				return null;
			}
			RealMatrix kthClassMatrix = new Array2DRowRealMatrix(orderedClassInstances[k], x_matrix.getColumnDimension());
			
			for(int i=0, row=0; i< totalNrOfInstances; i++)	{
				if(Math.round(y_matrix.getEntry(i, 0)) == Math.round(orderedClassLabels[k]))	{
					kthClassMatrix.setRowMatrix(row, x_matrix.getRowMatrix(i));
					row++;
				}
			}
			
			orderedClassMeans[k] = getMeanMatrix(kthClassMatrix);
			
		}
		
		return orderedClassMeans;
		
	}

	/**
	 * @param x_matrix
	 * @param y_matrix
	 * @param orderedClassLabels
	 * @param totalNrOfInstances
	 * @return
	 */
	protected static int[] getOrderedClassInstances(RealMatrix x_matrix,
			RealMatrix y_matrix, double[] orderedClassLabels, int totalNrOfInstances) {

		int[] orderedClassInstances = new int[orderedClassLabels.length];
		
		for(int k=0; k< orderedClassLabels.length; k++)	{
			int row = 0;
			for(int i=0; i< totalNrOfInstances; i++)	{
				if(Math.round(y_matrix.getEntry(i, 0)) == Math.round(orderedClassLabels[k]))	{
					row++;
				}
			}
			orderedClassInstances[k] = row;
		}
		return orderedClassInstances;
	}
	
	protected static double[] getClassPriorProbs(int totalNrOfInstances, int[] orderedClassInstances)	{
		
		int nrOfClasses = orderedClassInstances.length;
		double[] orderedClassPriorProbs = new double[nrOfClasses];
		
		for(int k=0; k< nrOfClasses; k++)	{
			orderedClassPriorProbs[k] = orderedClassInstances[k]/(double) totalNrOfInstances; 	
		}
		 
		return orderedClassPriorProbs;
	}
	
}
