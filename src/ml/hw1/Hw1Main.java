/**
 * 
 */
package ml.hw1;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;

import ml.hw1.InputOutput;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 26, 2014 [Sousa]
 */

public class Hw1Main {

//	protected static final double[] spam2ClassLabels = {0,1};
//	protected static final double[] mnist4ClassLabels = {1,3,7,8};
	protected static double[] givenClassLabels;
	protected static RealMatrix xMatrix;
	protected static RealMatrix yMatrix;
//	protected static int nrOfClassLabels;
	
	protected static final String INPUT_DIR = "src/main/resources/MlHw1Data/spam.csv";

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {

		//Intentionally kept blank
		
		//Initialize data instances
		/*List<String> instances = new InputOutput().initInstances(INPUT_DIR);
		
		//Calculate w* from Fisher's Discriminant
		int totalNrOfInstances = instances.size();
		
		//Calculate class means and covariances
		int[] orderedClassInstances = MatrixOperators.getOrderedClassInstances(xMatrix, yMatrix, spam2ClassLabels, totalNrOfInstances);
		RealMatrix[] orderedClassMeans = MatrixOperators.getOrderedClassMeans(xMatrix, yMatrix, spam2ClassLabels, totalNrOfInstances, orderedClassInstances);
		RealMatrix withinClassCovariance = FisherDiscriminant.getWithinClassCovariance(xMatrix, yMatrix, orderedClassMeans, 
				totalNrOfInstances, xMatrix.getColumnDimension(), spam2ClassLabels);
		
		RealMatrix w = FisherDiscriminant.get2ClassFishersDiscriminant(withinClassCovariance, orderedClassMeans[0], orderedClassMeans[1]);
		InputOutput.printMatrix(w, "Fisher'sW_Matrix");
		
		System.out.println("Test complete");*/

	}

}
