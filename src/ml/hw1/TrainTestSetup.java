/**
 * 
 */
package ml.hw1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 27, 2014 [Sousa]
 */
public class TrainTestSetup {

	protected static String[] createNFoldCVSetup(int totalNrOfInstances, int nrOfFolds)	{
		
		if(totalNrOfInstances < nrOfFolds)	{
			System.err.println("Please set number of folds >= number of instances");
			return null;
		}
		
		Map<Integer, Integer> selectedInstanceIds = new HashMap<Integer, Integer>();
		//int remainingInstances = totalNrOfInstances;
		
		String[] foldInstances = new String[nrOfFolds];
		
		for(int foldNumber=nrOfFolds; foldNumber>0; foldNumber--)	{
			int nrOfFilesInThisFold = (totalNrOfInstances- selectedInstanceIds.size())/foldNumber;
			
			foldInstances[nrOfFolds-foldNumber] = "";
			for(int i=0; i< nrOfFilesInThisFold;)	{
				int randomNumber = new Random().nextInt(totalNrOfInstances);
				if(!selectedInstanceIds.containsKey(randomNumber))	{
					selectedInstanceIds.put(randomNumber, 1);
					foldInstances[nrOfFolds-foldNumber] += randomNumber + " ";
					i++;
				}
			}
		}
		
		return foldInstances;
	}
	
	protected static String createTrainTestSetup(int totalNrOfInstances, double trainPercentage)	{
		
		Map<Integer, Integer> selectedInstanceIds = new HashMap<Integer, Integer>();
		double testPercentage = 100- trainPercentage;
		int nrOfFilesInTestSet = (int) testPercentage*totalNrOfInstances/100;
		
		String testInstances = "";
		
		for(int i=0; i< nrOfFilesInTestSet;)	{
			int randomNumber = new Random().nextInt(totalNrOfInstances);
			if(!selectedInstanceIds.containsKey(randomNumber))	{
				selectedInstanceIds.put(randomNumber, 1);
				testInstances += randomNumber + " ";
				i++;
			}
		}
		System.out.println(nrOfFilesInTestSet + " Instances in test set:\n");
		System.out.println(testInstances);
		
		return testInstances;
	}

	/**
	 * @param testSets
	 * @param foldNr
	 * @return
	 */
	protected static List<Integer> getTestInstanceIds(String[] testSets, int foldNr) {

		String testSet = testSets[foldNr];
		
		String[] instanceIdStrings = testSet.split(" ");
		List<Integer> instanceIds = new ArrayList<Integer>();
		
		for(int id=0; id< instanceIdStrings.length; id++)	{
			instanceIds.add(Integer.parseInt(instanceIdStrings[id]));
		}
		
		return instanceIds;
	}

	/**
	 * @param x_matrix
	 * @param y_matrix
	 * @param testInstanceIds
	 * @return
	 */
	protected static RealMatrix[] getTrainTestMatrices(RealMatrix x_matrix,
			RealMatrix y_matrix, List<Integer> testInstanceIds) {
		
		int totalNrOfInstances = y_matrix.getRowDimension();
		int totalNrOfAttributes = x_matrix.getColumnDimension();
		
		int nrOfTestInstances = testInstanceIds.size();
		int nrOfTrainInstances = totalNrOfInstances - nrOfTestInstances;
		
		RealMatrix[] trainTestMatrices = new RealMatrix[4];
		
		int trainX = 0;
		int trainY = 1;
		int testX = 2;
		int testY = 3;
		
		trainTestMatrices[trainX] = new Array2DRowRealMatrix(nrOfTrainInstances, totalNrOfAttributes);
		trainTestMatrices[trainY] = new Array2DRowRealMatrix(nrOfTrainInstances, 1);
		trainTestMatrices[testX] = new Array2DRowRealMatrix(nrOfTestInstances, totalNrOfAttributes);
		trainTestMatrices[testY] = new Array2DRowRealMatrix(nrOfTestInstances, 1);
		
		for(int id=0, trainId=0, testId=0; id< totalNrOfInstances; id++)	{
			if(testInstanceIds.contains(id))	{
				trainTestMatrices[testX].setRowMatrix(testId, x_matrix.getRowMatrix(id));
				trainTestMatrices[testY].setRowMatrix(testId, y_matrix.getRowMatrix(id));
				testId++;
			}
			else	{
				trainTestMatrices[trainX].setRowMatrix(trainId, x_matrix.getRowMatrix(id));
				trainTestMatrices[trainY].setRowMatrix(trainId, y_matrix.getRowMatrix(id));
				trainId++;
			}
		}
		
		return trainTestMatrices;
	}

	/**
	 * @param testInstances
	 * @param totalNrOfInstances
	 * @param train_Percent
	 * @return
	 */
	protected static String getTrainInstances(String testInstances,
			int totalNrOfInstances, double train_Percent) {

		Map<Integer, Integer> selectedInstanceIds = new HashMap<Integer, Integer>();
		int nrOfTestInstances = testInstances.split(" ").length;
		int nrOfFilesInTrainSet = (int) train_Percent*(totalNrOfInstances-nrOfTestInstances)/100;
		
		String trainInstances = "";
		
		for(int i=0; i< nrOfFilesInTrainSet;)	{
			int randomNumber = new Random().nextInt(totalNrOfInstances);
			if(!selectedInstanceIds.containsKey(randomNumber) && 
					(!testInstances.contains(randomNumber+" ") || !testInstances.contains(" "+ randomNumber)))	{
				selectedInstanceIds.put(randomNumber, 1);
				trainInstances += randomNumber + " ";
				i++;
			}
		}
		System.out.println(nrOfFilesInTrainSet + " Instances in train set:\n");
		//System.out.println(trainInstances);
		
		return trainInstances;
	}

	/**
	 * @param yMatrix
	 * @param selectedInstances
	 * @return
	 */
	protected static RealMatrix getSelectedYMatrix(RealMatrix yMatrix,
			String selectedInstances) {

		String[] selectedInstanceIds = selectedInstances.split(" ");
		int nrOfSelectedInstances = selectedInstanceIds.length;
		RealMatrix selectedYMatrix = new Array2DRowRealMatrix(nrOfSelectedInstances,1);
		
		for(int row=0; row< nrOfSelectedInstances; row++)	{
			selectedYMatrix.setRowMatrix(row,yMatrix.getRowMatrix(Integer.parseInt(selectedInstanceIds[row])));
		}
		
		return selectedYMatrix;
	}

	/**
	 * @param xMatrix
	 * @param selectedInstances
	 * @return
	 */
	protected static RealMatrix getSelectedXMatrix(RealMatrix xMatrix,
			String selectedInstances) {

		String[] selectedInstanceIds = selectedInstances.split(" ");
		int nrOfSelectedInstances = selectedInstanceIds.length;
		RealMatrix selectedXMatrix = new Array2DRowRealMatrix(nrOfSelectedInstances,xMatrix.getColumnDimension());
		
		for(int row=0; row< nrOfSelectedInstances; row++)	{
			selectedXMatrix.setRowMatrix(row,xMatrix.getRowMatrix(Integer.parseInt(selectedInstanceIds[row])));
		}
		
		return selectedXMatrix;
	}
	
}
