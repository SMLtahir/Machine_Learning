package ml.hw2;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

public class mysgdsvm {

	 protected static String INPUT_FILENAME;				// = "src/ml/hw2/MNIST-13.csv";
	 protected static int K;								// = 2000;
	 protected static int NR_OF_RUNS;						// = 5;
	 
	 protected static final double LAMBDA = 0.5;
	 protected static RealMatrix yMatrix;
	 protected static RealMatrix xMatrix;
	 protected static final int[] T_Array = {30000};
	 
	public static void main(String[] args) throws IOException {

		//Get from user input
		INPUT_FILENAME = args[0];
		if(Integer.parseInt(args[1]) == 1)
			K =1;
		else
			K = Integer.parseInt(args[1])/2;
		NR_OF_RUNS = Integer.parseInt(args[2]);
		
		//Initialize data instances
		List<String> instances = new InputOutput().initInstances(INPUT_FILENAME);
		yMatrix = HwMain.yMatrix;
		xMatrix = HwMain.xMatrix;
		int totalNrOfInstances = yMatrix.getRowDimension();
		int totalNrOfAttributes = xMatrix.getColumnDimension();	
		double[] timeElapsedInRun = new double[NR_OF_RUNS];
		double averageTimePerRun = 0;
		double averagePrimalFunction = 0.0;
		String outString = "";
		
			for(int run = 0; run< NR_OF_RUNS; run++)	{
				long startTime = System.nanoTime();
				
				//This is for testing for different values of T, if required (currently not used)
				for(int t_run = 0; t_run< T_Array.length; t_run++)	{
					int T = T_Array[t_run];	
						
					//Initialize wMatrix
					RealMatrix[] wMatrix = new RealMatrix[2];

					//Initialize wMatrix[0]
					wMatrix[0] = new Array2DRowRealMatrix(totalNrOfAttributes, 1);
					
					double eta_t = 0.0;
					
					//Loop over T iterations
					int t = 0;
					boolean startFlag = false;
					for(t=0; t< T; t++)	{										
						//Choose a subset, A_t of training set S
						String batchString = getBatch(totalNrOfInstances, K);
						
						if(startFlag)
							wMatrix[0] = wMatrix[1];
						else
							startFlag = true;
						
						//Loop whole batch to find A_t+
						String aTPosBatchString = getATPos(batchString, wMatrix[0]);
						String[] aTPosInstanceIds = aTPosBatchString.split(" ");
						
						//Set ETA_t
						eta_t = 1.0/(LAMBDA*(double)(t+1));
						
						RealMatrix tempW = wMatrix[0].scalarMultiply((1-eta_t*LAMBDA));
						for(int i =0; i< aTPosInstanceIds.length; i++)	{
							if(aTPosInstanceIds[i].equals(""))	continue;
							int instanceId = Integer.parseInt(aTPosInstanceIds[i]);
							int yValue =  (yMatrix.getEntry(instanceId, 0)==1) ? 1 : -1;
							RealMatrix x_i = xMatrix.getRowMatrix(instanceId);
							
							tempW = tempW.add(x_i.scalarMultiply( (double)(yValue)* eta_t/(double)K).transpose() );
						}
						
						wMatrix[1] = tempW.scalarMultiply(
								Math.min(
										1, (1/Math.sqrt(LAMBDA)/(tempW.getNorm()) )));
						
						int yValue;
						double primalFunction = (LAMBDA/2)*Math.pow(wMatrix[1].getNorm(),2);
						for(int i=0; i< totalNrOfInstances; i++)	{
							yValue = (yMatrix.getEntry(i, 0)==1) ? 1: -1;
							primalFunction += (1.0/(double)totalNrOfInstances)*
									Math.max( 0, 1- wMatrix[1].transpose().getRowVector(0).dotProduct(xMatrix.getRowVector(i) )*(double)(yValue) );
							
						}
						System.out.println((run+1)+ "\t"+ t+ "\t"+ primalFunction+ "\t"+K);
						outString += (run+1)+ "\t"+ t+ "\t"+ primalFunction+ "\t"+K + "\n";
					
					}
					timeElapsedInRun[run] = (double)(System.nanoTime() - startTime)/1000000;
					averageTimePerRun = averageTimePerRun+ timeElapsedInRun[run]; 
//					
					System.out.println("Run: "+ (run+1)+ "\nTime elapsed(ms): "+ timeElapsedInRun[run]);
					outString += "\n"+ "Run: "+ (run+1)+ "\nTime elapsed(ms): "+ timeElapsedInRun[run] + "\n";
				}

			}

			averageTimePerRun = averageTimePerRun/NR_OF_RUNS;
			averagePrimalFunction = averagePrimalFunction/(double)NR_OF_RUNS;
			StandardDeviation stdDev = new StandardDeviation();
			double timeStdDev = stdDev.evaluate(timeElapsedInRun);
			System.out.println("Avg Time per run (ms): "+ averageTimePerRun+ ", Std Dev. in time (ms): "+ timeStdDev);
			outString +=  "\n"+ "Avg Time per run (ms): "+ averageTimePerRun+ ", Std Dev. in time (ms): "+ timeStdDev + "\n";
			
			//Write to file
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("src/ml/hw2/" + K+"K_sgdsvmOutput.txt"),"UTF-8"));
			writer.write(outString);
			writer.close();			
	}

	protected static String getATPos(String batchInstancesString, RealMatrix tthWMatrix) {

		String aTPosBatchString = "";
		String[] batchInstances = batchInstancesString.split(" ");
		double error = 0.0;
		int yValue;
		int instanceId = -1;
		for(int i=0; i< K; i++)	{
			instanceId = Integer.parseInt(batchInstances[i]);
			yValue = (int) ((yMatrix.getEntry(instanceId, 0)==1) ? 1 : -1);
			RealVector x_i = xMatrix.getRowVector(instanceId);
			RealVector w = tthWMatrix.transpose().getRowVector(0);
			
			error = 1 - (double)(yValue)*(w.dotProduct(x_i));
			if(error > 0)
				aTPosBatchString += instanceId + " ";
			
		}
		
		return aTPosBatchString;
	}

	protected static String getBatch(int totalNrOfInstances, double batchSize) {

		Map<Integer, Integer> selectedInstanceIds = new HashMap<Integer, Integer>();
		
		String batchInstances = "";
		
		for(int i1=0, i2=0; ;)	{
			int randomNumber = new Random().nextInt(totalNrOfInstances);
			int yValue = (yMatrix.getEntry(randomNumber, 0)==1) ? 1: -1;
			if(!selectedInstanceIds.containsKey(randomNumber))	{
				if(yValue == 1)	{
					if(i1< K)	{
						selectedInstanceIds.put(randomNumber, 1);
						batchInstances += randomNumber + " ";		
					}
					i1++;
				}
				else if(yValue == -1)	{
					if(i2< K)	{
						selectedInstanceIds.put(randomNumber, 1);
						batchInstances += randomNumber + " ";
						i2++;
					}
				}
				
			}
			if(i1 >= K && i2 >= K)	break;
		}
//		System.out.println(batchSize + " Instances in batch:\n");
		
		return batchInstances;
	}

}
