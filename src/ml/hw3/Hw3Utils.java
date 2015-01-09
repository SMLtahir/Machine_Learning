package ml.hw3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class Hw3Utils {

	protected static RealMatrix[] getBootstrappedSample(int totalNrOfInstances, RealMatrix trainXMatrix, RealMatrix trainYMatrix) {

		List<Integer> sample = new ArrayList<Integer>();
		RealMatrix[] xyMatrix = new Array2DRowRealMatrix[2];
		RealMatrix newXMatrix = new Array2DRowRealMatrix(trainXMatrix.getRowDimension(), trainXMatrix.getColumnDimension());
		RealMatrix newYMatrix = new Array2DRowRealMatrix(trainYMatrix.getRowDimension(), trainYMatrix.getColumnDimension());
		
		for(int i=0; i< totalNrOfInstances; i++)	{
			int randomNumber = new Random().nextInt(totalNrOfInstances);
			sample.add(randomNumber);
			newXMatrix.setRow(i, trainXMatrix.getRow(randomNumber));
			newYMatrix.setRow(i, trainYMatrix.getRow(randomNumber));
		}
		
		xyMatrix[0] = newXMatrix;
		xyMatrix[1] = newYMatrix;
		return xyMatrix;
	}
	
	protected static int getBaggedPrediction(
			RealMatrix[] baseClassifierParams, int nrOfBaseClassifiers, RealMatrix testXMatrix, int instanceId) {

		int class0Votes = 0;
		int class1Votes = 0;
		
		for(int i=0; i< nrOfBaseClassifiers; i++)	{
			int level0Attri = (int) baseClassifierParams[i].getEntry(0, 1);
//			System.out.println("Attribute on root: "+ level0Attri);
			double test0AttriValue = testXMatrix.getEntry(instanceId, level0Attri);
			int level1Attri = -1;
			int predictedLabel = 0;
			
			for(int j=0; j< baseClassifierParams[i].getRowDimension(); j++)	{
				if(baseClassifierParams[i].getEntry(j, 2) == test0AttriValue)	{
					level1Attri = (int) baseClassifierParams[i].getEntry(j, 3);
//					System.out.print("level1Attri: "+ level1Attri);
				}
			}
//			System.out.println("Attribute on level 1: "+ level1Attri);
			if(level1Attri == -1)	{
				for(int j=0; j< baseClassifierParams[i].getRowDimension(); j++)	{
					if(baseClassifierParams[i].getEntry(j, 0) == 0)	{
						break;
					}
					if(baseClassifierParams[i].getEntry(j, 2) == test0AttriValue)	{
						predictedLabel = (int) baseClassifierParams[i].getEntry(j, 5);
					}
				}
			}
			else	{
				double test1AttriValue = testXMatrix.getEntry(instanceId, level1Attri);
				for(int j=0; j< baseClassifierParams[i].getRowDimension(); j++)	{
					if(baseClassifierParams[i].getEntry(j, 0) == 0)	{
						break;
					}
					if(baseClassifierParams[i].getEntry(j, 2) == test0AttriValue &&
							baseClassifierParams[i].getEntry(j, 3) == level1Attri &&
							baseClassifierParams[i].getEntry(j, 4) == test1AttriValue)	{
						predictedLabel = (int) baseClassifierParams[i].getEntry(j, 5);
					}
				}	
			}
			

			if(predictedLabel == 0)
				class0Votes++;
			else
				class1Votes++;
		}
		if(class0Votes > class1Votes)
			return (-1);
		else
			return (1);
	}

	protected static ArrayList<Integer> getRestrictedIds(int totalNrOfAttributes,
			int nrOfRestrictedIds) {

		ArrayList<Integer> restrictedIds = new ArrayList<Integer>();
		for(int i=0; i< nrOfRestrictedIds;)	{
			int randomNr = new Random().nextInt(totalNrOfAttributes);
			if(!restrictedIds.contains(randomNr))	{
				restrictedIds.add(randomNr);
				i++;
			}
				
		}
		
		return restrictedIds;
	}
	
	
}
