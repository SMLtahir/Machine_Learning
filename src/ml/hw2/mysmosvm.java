package ml.hw2;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

public class mysmosvm {

//	public static final String INPUT_FILENAME = "src/ml/hw2/MNIST-13.csv";
//	public static final int MAX_RUNS = 5;
	public static final int ITERATIONS = 5;

	//Input constants
	protected static final double C = 0.5;
	protected static final double TOL = 0.001;
	protected static final double EPS = 0.001;
	protected static double[] alpha;
	protected static RealMatrix yMatrix;
	protected static RealMatrix xMatrix;
	protected static RealMatrix wMatrix;
	protected static double b;
	protected static int totalNrOfInstances;
	protected static int totalNrOfAttributes;
	protected static Map<Integer, Double> errorCache;
	
	public static void main(String[] args) throws IOException {
		
		String INPUT_FILENAME = args[0];
		int MAX_RUNS = Integer.parseInt(args[1]);

		//Initialize data instances
		List<String> instances = new InputOutput().initInstances(INPUT_FILENAME);
		yMatrix = HwMain.yMatrix;
		xMatrix = HwMain.xMatrix;
		totalNrOfInstances = yMatrix.getRowDimension();
		totalNrOfAttributes = xMatrix.getColumnDimension();
		
		double[] timeElapsedInRun = new double[MAX_RUNS];
		long averageTimePerRun = 0;
//		String outString = "";
		for(int run=0; run< MAX_RUNS; run++)	{
			long startTime = System.nanoTime();	
			wMatrix = new Array2DRowRealMatrix(totalNrOfAttributes, 1);
			errorCache = new HashMap<Integer, Double>();
			
			//Initialize alpha, threshold b and nr of passes
			alpha = new double[totalNrOfInstances];
			b = 0.0;
			int numChanged = 0;
			int examineAll = 1;
			int nrOfIterations = 0;
			initializeErrorCache();
			
			while(numChanged > 0 || examineAll == 1)	{
				numChanged = 0;
				if (examineAll == 1)	{
					
					//loop I over all training examples
					for (int i=0; i< totalNrOfInstances; i++)	{
						numChanged += examineExample(i);	
					}
				}
				else
					//loop I over examples where alpha is not 0 & not C
					for(int i=0; i< totalNrOfInstances; i++)	{
						if(alpha[i] > 0 && alpha[i] < C)	{
							numChanged += examineExample(i);		
						}
					}
				
					//Alternate between iterating over whole training set vs iterating over only non-bound samples
					if(examineAll == 1)
						examineAll = 0;
					else if (numChanged == 0)
						examineAll = 1;
					
					//Check dual objective function value
					double dualObjFn = getDualObjFunc();
					System.out.println("Dual objective function: "+ dualObjFn);
//					outString += "Dual objective function: "+ dualObjFn+ "\n";
					
					nrOfIterations++;
//					if(nrOfIterations >= ITERATIONS)
//						break;
					
			}
			
			
			System.out.println("Test Complete");
			
			//Find accuracy (just for interest)
			/*double accuracy = 0.0;
			int nrOfCorrectPredictions = 0;
			for(int i=0; i< totalNrOfInstances; i++)	{
				double ithActualValue = (yMatrix.getEntry(i, 0)==1)? 1: -1;
				double prediction = getHypothesisAtIndex(i);
				if(prediction < 0)	
					prediction = -1;
				else
					prediction = 1;
				
				 if(ithActualValue == prediction)
					 nrOfCorrectPredictions++;
			}
			accuracy = 100* (double) nrOfCorrectPredictions/ (double)totalNrOfInstances;
			System.out.println("Accuracy: "+ accuracy);*/
		
			timeElapsedInRun[run] = (System.nanoTime() - startTime)/1000000;
			averageTimePerRun += timeElapsedInRun[run]; 
			System.out.println("Time elapsed (ms) in Run "+ (run+1)+ ": "+ timeElapsedInRun[run]);
//			outString += "\n"+ "Time elapsed (ms) in Run "+ (run+1)+ ": "+ timeElapsedInRun[run] + "\n";
		}
		averageTimePerRun = averageTimePerRun/MAX_RUNS;
		StandardDeviation stdDev = new StandardDeviation();
		double timeStdDev = stdDev.evaluate(timeElapsedInRun);
		System.out.println("Avg Time per run (ms): "+ averageTimePerRun+ ", Std Dev. in time (ms): "+ timeStdDev);
//		outString += "\n"+ "Avg Time per run (ms): "+ averageTimePerRun+ ", Std Dev. in time (ms): "+ timeStdDev + "\n";
		
//		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("src/ml/hw2/" + "smoOutput.txt"),"UTF-8"));
//		writer.write(outString);
//		writer.close();
	}
	
	private static double getDualObjFunc() {

		double dualObjFunction = 0.0;
		double sumOfAlpha = 0.0;
		double sumOfKernel = 0.0;
		double yiValue, yjValue = 0.0;
		double xDotProd = 0.0;
		RealMatrix diagY = getDiagY();
		RealMatrix alphaMatrix = new Array2DRowRealMatrix(yMatrix.getRowDimension(), 1);
		alphaMatrix.setColumn(0, alpha);
		
		for(int i=0; i< totalNrOfInstances; i++)	{
			
			yiValue = (yMatrix.getEntry(i, 0)==1)? 1: -1;
			sumOfAlpha += alpha[i];
			
		}
		//Use matrix form to calculate dual objective function
		dualObjFunction = sumOfAlpha - alphaMatrix.transpose().multiply(diagY).scalarMultiply(xDotProd).multiply(diagY).multiply(alphaMatrix).scalarMultiply(0.5).getEntry(0, 0);
		
		return dualObjFunction;
	}

	private static RealMatrix getDiagY() {

		//Build a diagonal square matrix of all the entries of Y
		RealMatrix diagY = new Array2DRowRealMatrix(yMatrix.getRowDimension(), yMatrix.getRowDimension());
		for(int i=0; i< yMatrix.getRowDimension(); i++)	{
			diagY.setEntry(i, i, yMatrix.getEntry(i, 0));
		}
		
		return diagY;
	}

/*	private static void updateErrorCache() {

		for(int i=0; i< totalNrOfInstances; i++)	{
			if(alpha[i] > 0 && alpha[i] < C)	{
				double error = getPredictionErrorAtIndex(i);
				if(error != 0)
					errorCache.put(i, error);	
			}
			
		}
		
	}*/

	private static void initializeErrorCache() {

		for(int i=0; i< totalNrOfInstances; i++)	{
			double error = getPredictionErrorAtIndex(i);
			if(error !=0)
				errorCache.put(i, error);
		}
	}

	protected static int examineExample(int i2) {

		double y2=0; 
		double alph2=0, E2=0, r2=0;
		Set<Integer> errorKeyset = errorCache.keySet();
		List<Integer> errorKeyList = new ArrayList<Integer>(errorKeyset);
		
		y2 = (yMatrix.getEntry(i2, 0) == 1)? 1: -1;
		alph2 = alpha[i2];
		
		E2 = getPredictionErrorAtIndex(i2);
		errorCache.put(i2, E2);
 
		r2 =  E2*(y2);
		
		if ( (r2 < -TOL && alph2 < C) || (r2 > TOL && alph2 > 0) )	{
			//Option 1- Get perfect i1 from 2nd heuristic
			{
				int i1=0;
				double tmax=0;
      
				//Choose i1
				i1 = -1; 
				tmax = 0;
				for (int k : errorKeyset)	{
					if (alpha[k] > 0 && alpha[k] < C) 
					{
						double E1=0, temp=0;
      
						E1 = errorCache.get(k);
						temp = Math.abs(E2 - E1);
						if (temp > tmax)
						{
							tmax = temp;
							i1 = k;
      
						}
					}
   
				}
				if (i1 >= 0) 
				{
					//Check for valid step
					if (takeStep(i1, i2)==1)
						return 1;
				}
			}
			//Option 2- choose random index and iterate over errorCache to find perfect i1
			{
				if(errorKeyList.size()!=0)	{
					int randomIndex = new Random().nextInt(errorKeyList.size());
					int iterEnd = errorKeyList.size();
					boolean flag = false;
					
					for (int index = randomIndex; index< iterEnd; index++) 
					{
						int i1 = errorKeyList.get(index);
	        
						//If i1 is non-bounded
						if (alpha[i1] > 0 && alpha[i1] < C) 
						{
							//Check for valid step
							if (takeStep(i1, i2)==1)
								return 1;
						}
						//Start from 0 to randomIndex
						if(index+1 == iterEnd && flag == false)	{
							iterEnd = randomIndex;
							index = 0;
							flag = true;
						}
							
					}	
				}
				
			}

			//Option 3- take random i1 from errorCache
			{      
				if(errorKeyList.size()!=0)	{
					int randomIndex = new Random().nextInt(errorKeyList.size());
					int iterEnd = errorKeyList.size();
					boolean flag = false;
					for (int index = randomIndex; index< errorKeyList.size(); index++)  
					{
						int i1 = errorKeyList.get(index);
						if (takeStep(i1, i2)== 1)
							return 1;
						
						//Continue looping from 0 to randomIndex
						if(index+1 == iterEnd && flag == false)	{
							iterEnd = randomIndex;
							index = 0;
							flag = true;
						}
					}
					
				}
				
			}
		}
		return 0;
	}

	protected static int takeStep(int i1, int i2)	{
		
		double a1=0, a2=0;       /* new values of alpha_1, alpha_2 */
		double E1=0, E2=0, L=0, H=0, Lobj=0, Hobj=0;
		double alph1 = alpha[i1];		//Old value
		double alph2 = alpha[i2];		//Old value
		double y1 = (yMatrix.getEntry(i1, 0)==1)? 1: -1;
		double y2 = (yMatrix.getEntry(i2, 0)==1)? 1: -1;
		double s = (double)(y1)*(double)(y2);			
		
		if(i1 == i2) 
			return 0;

		E1 = getPredictionErrorAtIndex(i1);
		errorCache.put(i1, E1);
		
			E2 = errorCache.get(i2);

		//Use these values of L and H if ithActualValue != jthActualValue
		if(y1 != y2)	{
			L = Math.max(0, alpha[i2]- alpha[i1]);
			H = Math.min(C, C+ alpha[i2]- alpha[i1]);
		}
		
		//Use these values of L and H if ithActualValue == jthActualValue					
		else{
			L = Math.max(0, alpha[i1]+ alpha[i2] - C);
			H = Math.min(C, alpha[i1]+ alpha[i2]);	
		}

		if (L == H)
			return 0;
				
		RealVector ithVector = xMatrix.getRowVector(i1);
		RealVector jthVector = xMatrix.getRowVector(i2);
		
		double ETA = 2*ithVector.dotProduct(jthVector) 
				- ithVector.dotProduct(ithVector) 
				- jthVector.dotProduct(jthVector);

		if(ETA < 0)	{
			
			a2 = alph2 - (double)(y2)*(E1-E2)/ETA;
			if (a2 < L) 
				a2 = L;
			else if (a2 > H) 
				a2 = H;
		}
		else	{
//			Lobj = objective function at a2=L
//			Hobj = objective function at a2=H
	
			Lobj = getObjAt(L, i2);
			Hobj = getObjAt(H, i2);
			
			if (Lobj > Hobj+EPS)
				a2 = L;
			else if (Lobj < Hobj-EPS)
				a2 = H;
			else
				a2 = alph2;			
		}
		if (a2 < 0.00000001)
			a2 = 0;
		else if (a2 > C-0.00000001)
			a2 = C;
			
		if (Math.abs(a2-alph2) < EPS*(a2+alph2+EPS) )
			return 0;
		a1 = alph1+s*(alph2-a2);
		
//				Update threshold to reflect change in Lagrange multipliers
				double b1 = E1 
						+ (double)(y1)*(a1 - alph1)*ithVector.dotProduct(ithVector)
						+ (double)(y2)*(a2 - alph2)*ithVector.dotProduct(jthVector)+ b;
				
				double b2 = E2 
						+ (double)(y1)*(a1 - alph1)*ithVector.dotProduct(jthVector)
						+ (double)(y2)*(a2 - alph2)*jthVector.dotProduct(jthVector)+ b;
				
				double b_old = b;

				//Update b
				if(a1 > 0 && a1 < C)
					b = b1;
				else if(a2 > 0 && a2 < C)
					b = b2;
				else
					b = (b1+b2)/2;
				
				double deltaB = b - b_old;
//				Update weight vector to reflect change in a1 & a2, if linear SVM
				wMatrix = wMatrix.add(xMatrix.getRowMatrix(i1).scalarMultiply((a1-alph1)*(double)(y1)).transpose());
				wMatrix = wMatrix.add(xMatrix.getRowMatrix(i2).scalarMultiply((a2-alph2)*(double)(y2)).transpose());
				
//				Update error cache using new Lagrange multipliers	  
				for (int index : errorCache.keySet())	{
					if(index == i1 || index == i2)
						continue;
					if (alpha[index] > 0 && alpha[index] < C)
					{  
						double temp = errorCache.get(index);
						temp +=  (double)(y1)*(a1 - alph1)*ithVector.dotProduct(jthVector) 
								+ (double)(y2)*(a2 - alph2)*ithVector.dotProduct(jthVector)- deltaB;
						errorCache.put(index, temp);
					}
				}
					
				if(a1 != 0 && a1!= C)
					errorCache.remove(i1);
				if(a2 != 0 && a2!= C)
					errorCache.remove(i2);
				
//				Store a1 in the alpha array
				alpha[i1] = a1;
				
//				Store a2 in the alpha array
				alpha[i2] = a2;
				
				return 1;
	}
	
	private static double getObjAt(double a2, int i2) {

		double dualObjFunction = 0.0;
		double sumOfAlpha = 0.0;
		double sumOfKernel = 0.0;
		double yiValue, yjValue = 0.0;
		double xDotProd = 0.0;
		
		for(int i=0; i< totalNrOfInstances; i++)	{
			double alph1 = (i==i2)? a2: alpha[i];
			
			yiValue = (yMatrix.getEntry(i, 0)==1)? 1: -1;
			sumOfAlpha += alph1;
			
			for(int j=0; j< totalNrOfInstances; j++)	{
				double alph2 = (j==i2)? a2: alpha[j];
				yjValue = (yMatrix.getEntry(j, 0)==1)? 1: -1;
				xDotProd = xMatrix.getRowVector(i).dotProduct(xMatrix.getRowVector(j));
				sumOfKernel += (-0.5)*yiValue*yjValue*xDotProd*alph1*alph2;
			}
			
		}
		dualObjFunction = sumOfAlpha + sumOfKernel;
		
		return dualObjFunction;
	}

	protected static int isAlphaContainsZeroC(double[] alpha) {

		int nonZeroCCount = 0;
		for (int i=0; i< alpha.length; i++)	{
			if(alpha[i] != 0 && alpha[i] != C)	{
				nonZeroCCount++;
			}
		}
		return nonZeroCCount;
	}
	
	protected static double getHypothesisAtIndex(int index)	{
		
		
		double hypothesisAtIndex = Math.signum(wMatrix.transpose().getRowVector(0).dotProduct(xMatrix.getRowVector(index)) - b);
		
		return hypothesisAtIndex;
	}
	
	protected static double getPredictionErrorAtIndex(int index)	{
		
		double ithActualValue = (yMatrix.getEntry(index, 0)==1)? 1 : -1;	//Binarize the output
		double ithHypothesis = getHypothesisAtIndex(index);
		
		double predictionError = ithHypothesis - ithActualValue;
		
		return predictionError;
	}

}
