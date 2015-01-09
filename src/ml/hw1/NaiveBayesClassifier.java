/**
 * 
 */
package ml.hw1;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 29, 2014 [Sousa]
 */
public class NaiveBayesClassifier {

	/**
	 * @param dataMatrix
	 * @param yMatrix
	 * @param classToWordMatrix 
	 * @param classLabels 
	 */
	protected static final double TRAIN_PERCENTAGE = 80;
	protected static int nrOfClasses = Hw2Main.news20ClassLabels.length;
	protected static void runMultiClass(RealMatrix dataMatrix, RealMatrix yMatrix, RealMatrix classToWordMatrix, 
			int[] classLabels, int nrOfSplits, double[] train_Percent) {

		int totalNrOfInstances = yMatrix.getRowDimension();		
		int nrOfTrainPercents = train_Percent.length;
		double splitAverageErrorRate = 0.0;
		
		for(int split =0; split< nrOfSplits; split++)	{
			System.out.println("Starting split...");
			String testInstancesString = TrainTestSetup.createTrainTestSetup(totalNrOfInstances, TRAIN_PERCENTAGE);
			RealMatrix testYMatrix = TrainTestSetup.getSelectedYMatrix(yMatrix, testInstancesString);
			String[] testInstanceIds = testInstancesString.split(" "); 
			int nrOfTestInstances = testInstanceIds.length;
			double[] splitErrorRate = new double[nrOfTrainPercents];
			double runAverageErrorRate = 0.0;
			
			for(int runNr =0; runNr< nrOfTrainPercents; runNr++)	{
				System.out.println("Starting run...");
				String trainInstances = TrainTestSetup.getTrainInstances(testInstancesString, totalNrOfInstances, train_Percent[runNr]);
				RealMatrix trainYMatrix = TrainTestSetup.getSelectedYMatrix(yMatrix, trainInstances);
				double[] logOfClassPriorProbs = getLogOfClassPriorProb(trainYMatrix, classLabels);
			
				int nrOfCorrectPredictions = 0;
				
				for(int i=0; i< nrOfTestInstances; i++)	{
					double[] logOfDocGivenClassProbs = getLogOfDocGivenClassProbs(Hw2Main.wordsInDoc, Integer.parseInt(testInstanceIds[i]), classToWordMatrix);
					double[] logOfClassGivenDocProbs = new double[nrOfClasses];
					double[] classGivenDocProbs = new double[nrOfClasses];
					
					for(int classId=0; classId< nrOfClasses; classId++)	{
						logOfClassGivenDocProbs[classId] = logOfDocGivenClassProbs[classId] + logOfClassPriorProbs[classId];
						if(logOfClassGivenDocProbs[classId]==0)
							classGivenDocProbs[classId] = 0;
						else
							classGivenDocProbs[classId] = Math.exp(logOfClassGivenDocProbs[classId]);		
						//The denominator is common in all comparisons and so not included in this formula. It is taken as a constant
						
					}
					
					int prediction = getPredictedClassLabel(classGivenDocProbs, classLabels);
					
					if(prediction == Math.round(testYMatrix.getEntry(i, 0)))	{
						nrOfCorrectPredictions++;
					}
				}
				double accuracy = (double) nrOfCorrectPredictions/(double) nrOfTestInstances;
				double runErrorRate = 1-accuracy;
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
	 * @param classGivenDocProbs
	 * @param classLabels
	 * @return
	 */
	protected static int getPredictedClassLabel(double[] classGivenDocProbs,
			int[] classLabels) {

		int prediction = -1;
		double tempMax = -1;
		for(int i=0; i< classLabels.length; i++)	{
			if(classGivenDocProbs[i] >= tempMax)	{
				tempMax = classGivenDocProbs[i];
				prediction = i+1;
			}
		}
		
		return prediction;
	}

	/**
	 * @param wordsInDoc
	 * @param docId 
	 * @param classToWordMatrix 
	 * @return
	 */
	protected static double[] getLogOfDocGivenClassProbs(String[] wordsInDoc, int docId, RealMatrix classToWordMatrix) {

		double[] logOfdocGivenClassProbs = new double[nrOfClasses];
		
		String[] wordIdStrings = wordsInDoc[docId-1].split("\t");
		for(int classId=1; classId <= nrOfClasses; classId++)	{
			for(String wIdString : wordIdStrings)	{
				int wordId = Integer.parseInt(wIdString);
				double wordGivenClassProb = getWordGivenClassProb(wordId, classId, classToWordMatrix);
				if(wordGivenClassProb == 0)	{
					logOfdocGivenClassProbs[classId-1] += 0;
				}
				else
					logOfdocGivenClassProbs[classId-1] += Math.log(wordGivenClassProb);	
			}
			
		}
		
		return logOfdocGivenClassProbs;
	}

	/**
	 * @param wordId
	 * @param classId
	 * @param classToWordMatrix 
	 * @return
	 */
	protected static double getWordGivenClassProb(int wordId, int classId, RealMatrix classToWordMatrix) {

		int totalWordsInClass = 0;
		int wordFrequencyInClass = 0;
		
		int rowNr = classId-1;
		for(int col=0; col< classToWordMatrix.getColumnDimension(); col++)	{
			if((int) classToWordMatrix.getEntry(rowNr, col) != 0)	{
				totalWordsInClass += classToWordMatrix.getEntry(rowNr, col);
			}
			if(wordId == col)	{
				wordFrequencyInClass += classToWordMatrix.getEntry(rowNr, col);
			}
		}
		double wordGivenClassProb = (double) wordFrequencyInClass/(double) totalWordsInClass;
		return wordGivenClassProb;
	}

	protected static double getClassConditionalProbs()	{
		
		
		
		return 0.0;
	}

	protected static double[] getLogOfClassPriorProb(RealMatrix yMatrix, int[] classLabels)	{
		
		double[] classPriorProbs = new double[classLabels.length];
		double[] logOfClassPriorProbs = new double[classLabels.length];
		int totalNrOfDocuments = yMatrix.getRowDimension();
		Map<Integer, Integer> classFrequency = new HashMap<Integer, Integer>();
		
		for(int labelCount=0; labelCount< classLabels.length; labelCount++)	{
			classFrequency.put(classLabels[labelCount]-1,0) ;
		}
		
		for(int row=0; row< totalNrOfDocuments; row++)	{
			int newFrequency = classFrequency.get((int) yMatrix.getEntry(row, 0)- 1) + 1; 
			classFrequency.put((int) yMatrix.getEntry(row, 0), newFrequency);
		}
		
		for(int labelCount=0; labelCount< classLabels.length; labelCount++)	{
			classPriorProbs[labelCount] = (double) classFrequency.get(labelCount)/ (double) totalNrOfDocuments;
			if(classPriorProbs[labelCount] == 0)
				logOfClassPriorProbs[labelCount] = 0;
			else
				logOfClassPriorProbs[labelCount] += Math.log(classPriorProbs[labelCount]);
			//System.err.println("Prior Prob of "+ labelCount+ ": "+ classPriorProbs[labelCount]);
		}
		
		return logOfClassPriorProbs;
	}
}
