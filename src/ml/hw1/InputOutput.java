/**
 * 
 */
package ml.hw1;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * 
 * @author Tahir Sousa
 * @version last updated: Sep 26, 2014 [Sousa]
 */
public class InputOutput {

	protected static String OUTPUT_DIR = "src/main/resources/MlHw1Data/";
	protected static final int NR_OF_CLASSES = 20;
//	protected static final int NR_OF_DOCS = 11269;
	protected static final int NR_OF_WORDS = 53975;
	
protected List<String> initInstances(String inputDir) throws IOException	{
		
		List<String> instances = new ArrayList<String>();
		instances = FileUtils.readLines(new File(inputDir));
		int nrOfInstances = instances.size();
		int nrOfAttributes = instances.get(0).split(",").length;
		Map<Double, Integer> classLabels = new TreeMap<Double, Integer>();
		
		RealMatrix dataMatrix = new Array2DRowRealMatrix(nrOfInstances, nrOfAttributes);
		RealMatrix xMatrix = new Array2DRowRealMatrix(nrOfInstances, nrOfAttributes-1);
		RealMatrix yMatrix = new Array2DRowRealMatrix(nrOfInstances, 1);
		
		int matrixRow = 0;
		
		for (String instance : instances)	{
			
			String[] instanceAttributes = instance.split(",");
			double[] instanceAttributeValues = new double[instanceAttributes.length];
			double instanceClass = Integer.parseInt(instanceAttributes[0]);
			
			yMatrix.setEntry(matrixRow, 0, instanceClass);
			classLabels.put(instanceClass, 0);
			
			for (int col=0; col< nrOfAttributes; col++)	{
				instanceAttributeValues[col] = Double.parseDouble(instanceAttributes[col]);
				//System.out.print(instanceAttributeValues[col]+ "\t");
			}
			dataMatrix.setRow(matrixRow, instanceAttributeValues);
			matrixRow++;
			
			//System.out.println("Nr. of attributes: "+ instanceAttributes.length);
			//System.out.println("Testing...");
		}
		System.out.println("Total instances: "+ nrOfInstances);
		System.out.println("Total attributes: "+ (nrOfAttributes-1));
		
		//System.out.println("Printing Data matrix... ");
		//printMatrix(dataMatrix, "dataMatrix");
		
		xMatrix = dataMatrix.getSubMatrix(0, nrOfInstances-1, 1, nrOfAttributes-1);
		
		//System.out.println("Printing X-matrix... ");
		//printMatrix(xMatrix, "x_matrix");
		Hw1Main.xMatrix = xMatrix;
		
		//System.out.println("Printing Y-matrix... ");
		//printMatrix(yMatrix, "y_matrix");
		Hw1Main.yMatrix = yMatrix;
		
		Hw1Main.givenClassLabels = new double[classLabels.keySet().size()];
		int labelCount = 0;
		for(double label: classLabels.keySet())	{
			Hw1Main.givenClassLabels[labelCount] = label;
			labelCount++;
		}
		
		return instances;
		
	}

	public static void printMatrix(RealMatrix matrix, String fileName) throws IOException	{
		
		String outputString = "";
		for(int row=0; row< matrix.getRowDimension(); row++)	{
			for(int col=0; col< matrix.getColumnDimension(); col++)	{
				System.out.print(matrix.getEntry(row, col) + "\t");
				outputString += matrix.getEntry(row, col) + "\t";
			}	
			//System.out.println("Progress: "+ row/matrix.getRowDimension());
			
			System.out.println();
			outputString += "\n";
		}
		
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(OUTPUT_DIR + fileName + ".csv"),"UTF-8"));
		
		writer.write(outputString);	
		writer.close();
		
	}
	
	protected static List<String> initNewsInstances(String dataInputDir, 
			String labelsInputDir) throws IOException	{
		List<String> dataInstances = new ArrayList<String>();
		List<String> labelsInstances = new ArrayList<String>();
		
		dataInstances = FileUtils.readLines(new File(dataInputDir));
		labelsInstances = FileUtils.readLines(new File(labelsInputDir));
		
		int nrOfInstances = dataInstances.size();
		int nrOfAttributes = dataInstances.get(0).split(",").length;
		
		RealMatrix dataMatrix = new Array2DRowRealMatrix(nrOfInstances, nrOfAttributes);
		RealMatrix xMatrix = new Array2DRowRealMatrix(nrOfInstances, nrOfAttributes-1);
		RealMatrix yMatrix = new Array2DRowRealMatrix(nrOfInstances, 1);
		
		int matrixRow = 0;
		
		for (String instance : dataInstances)	{
			
			String[] instanceAttributes = instance.split(",");
			double[] instanceAttributeValues = new double[instanceAttributes.length];
			double instanceClass = Integer.parseInt(instanceAttributes[0]);
			
			yMatrix.setEntry(matrixRow, 0, instanceClass);
			
			for (int col=0; col< nrOfAttributes; col++)	{
				instanceAttributeValues[col] = Double.parseDouble(instanceAttributes[col]);
				//System.out.print(instanceAttributeValues[col]+ "\t");
			}
			dataMatrix.setRow(matrixRow, instanceAttributeValues);
			matrixRow++;
			
			//System.out.println("Nr. of attributes: "+ instanceAttributes.length);
			//System.out.println("Testing...");
		}
		System.out.println("Total instances, m: "+ nrOfInstances);
		System.out.println("Total attributes, n: "+ nrOfAttributes);
		
		//System.out.println("Printing Data matrix... ");
		//printMatrix(dataMatrix, "dataMatrix");
		
		xMatrix = dataMatrix.getSubMatrix(0, nrOfInstances-1, 1, nrOfAttributes-1);
		
		//System.out.println("Printing X-matrix... ");
		//printMatrix(xMatrix, "x_matrix");
		Hw1Main.xMatrix = xMatrix;
		
		//System.out.println("Printing Y-matrix... ");
		//printMatrix(yMatrix, "y_matrix");
		Hw1Main.yMatrix = yMatrix;
		
//		System.out.println("Mean Matrix: ");
//		printMatrix(MatrixOperators.getMeanMatrix(xMatrix));
		
		return dataInstances;

	}

	/**
	 * @param dataInputDir
	 * @return
	 * @throws IOException 
	 */
	protected static RealMatrix[] initDataMatrix(String dataInputDir, RealMatrix yMatrix) throws IOException {

		List<String> dataInstances = new ArrayList<String>();
		dataInstances = FileUtils.readLines(new File(dataInputDir));
		RealMatrix[] usefulData = new RealMatrix[2];
		
		int dataRows = dataInstances.size();
		int dataColumns = dataInstances.get(0).split(",").length;

		//Data matrix
		usefulData[0] = new Array2DRowRealMatrix(dataRows, dataColumns);
		
		//Class to word Matrix
		usefulData[1] = new Array2DRowRealMatrix(NR_OF_CLASSES, NR_OF_WORDS);

		System.out.println("Start building data matrix...");
		int classId = -1;
		int docId = -1;
		int wordId = -1;
		int wordFrequency = -1;
		for(int row=0; row< dataRows; row++)	{
			classId = -1;
			docId = -1;
			wordId = -1;
			wordFrequency = -1;
			
			for(int col=0; col< dataColumns; col++)	{
				usefulData[0].setEntry(row, col, Double.parseDouble(dataInstances.get(row).split(",")[col]));
			}
			
			classId = (int) yMatrix.getEntry(Integer.parseInt(dataInstances.get(row).split(",")[0]) - 1, 0);
			docId = (int) Integer.parseInt(dataInstances.get(row).split(",")[0]);
			wordId = (int) Integer.parseInt(dataInstances.get(row).split(",")[1]);
			wordFrequency = (int) Integer.parseInt(dataInstances.get(row).split(",")[2]);

//			System.out.println("DocId: "+ docId+ "WordId: "+ wordId);
			Hw2Main.wordsInDoc[docId-1] += wordId + "\t";	
			
			usefulData[1].setEntry(classId-1, wordId-1, wordFrequency);
//			System.out.println("Progress: "+ 100* (double) row/ (double) dataRows + "%");	
		}
		System.out.println("Initializing data matrix Complete.");
		return usefulData;
	}
	
	protected static RealMatrix initLabelsMatrix(String labelsInputDir) throws IOException	{
		
		List<String> labelsInstances = new ArrayList<String>();
		labelsInstances = FileUtils.readLines(new File(labelsInputDir));
		Map<Double, Integer> classLabels = new TreeMap<Double, Integer>();
		
		int labelRows = labelsInstances.size();
		
		RealMatrix yMatrix = new Array2DRowRealMatrix(labelRows, 1);
		
		System.out.println("Initializing Y matrix...");
		for(int row=0; row< labelRows; row++)	{
			double label = Double.parseDouble(labelsInstances.get(row));
				yMatrix.setEntry(row, 0, label);
				classLabels.put(label, 0);
		}
		System.out.println("Initializing labels Complete.");
		
		Hw2Main.news20ClassLabels = new int[classLabels.keySet().size()];
		int labelCount = 0;
		for(double label: classLabels.keySet())	{
			Hw2Main.news20ClassLabels[labelCount] = (int) label;
			labelCount++;
		}
		return yMatrix;
		
	}

	/**
	 * @param nrOfDocs 
	 * @return
	 */
	protected static void initializeWordsInDoc(int nrOfDocs) {

		Hw2Main.wordsInDoc = new String[nrOfDocs];
		
		for(int i=0; i< nrOfDocs; i++)	{
			Hw2Main.wordsInDoc[i] = "";
		}
		
	}


}
