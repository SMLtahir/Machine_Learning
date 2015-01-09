/**
 * 
 */
package ml.hw2;

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

	protected static String OUTPUT_DIR = "src/ml/hw2/";
	protected static final int NR_OF_CLASSES = 20;
	
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
		HwMain.xMatrix = xMatrix;
		
		//System.out.println("Printing Y-matrix... ");
		//printMatrix(yMatrix, "y_matrix");
		HwMain.yMatrix = yMatrix;
		
		HwMain.givenClassLabels = new double[classLabels.keySet().size()];
		int labelCount = 0;
		for(double label: classLabels.keySet())	{
			HwMain.givenClassLabels[labelCount] = label;
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

}
