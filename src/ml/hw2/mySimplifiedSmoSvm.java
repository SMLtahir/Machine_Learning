package ml.hw2;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class mySimplifiedSmoSvm {

	public static final String INPUT_FILENAME = "src/ml/hw2/MNIST-13.csv";
	public static final int NR_OF_RUNS = 5;
	
	//Input constants
	protected static final double C = 10;
	protected static final double TOL = 0.00001;
	protected static final int MAX_PASSES = 5;
	
	public static void main(String[] args) throws IOException {
		
//		String INPUT_FILENAME = args[0];
//		int NR_OF_RUNS = Integer.parseInt(args[1]);

		//Initialize data instances
		List<String> instances = new InputOutput().initInstances(INPUT_FILENAME);
		RealMatrix yMatrix = HwMain.yMatrix;
		RealMatrix xMatrix = HwMain.xMatrix;
		int totalNrOfInstances = yMatrix.getRowDimension();
		
		//Initialize alpha, threshold b and nr of passes
		double[] alpha = new double[totalNrOfInstances];
		double b = 0.0;
		int passes = 0;
		
		while(passes < MAX_PASSES)	{
			
			int nrOfChangedAlphas = 0;
			for(int i =0; i< totalNrOfInstances; i++)	{
//				System.out.println("Iteration: "+ i+ " / "+ totalNrOfInstances);
				
				double ithPredictionError = getPredictionErrorAtIndex(alpha, yMatrix, xMatrix, b, i);
				double ithActualValue = (yMatrix.getEntry(i, 0)==1)? 1: -1;	//Binarize the output
				
				if( (ithActualValue*ithPredictionError < -TOL && alpha[i] < C)
						|| (ithActualValue*ithPredictionError > TOL && alpha[i] > 0) )	{
					
					//Select j != i
					int j = new Random().nextInt(totalNrOfInstances);
					while(j == i)	{
						j = new Random().nextInt(totalNrOfInstances);
					}
					
					double jthPredictionError = getPredictionErrorAtIndex(alpha, yMatrix, xMatrix, b, j);
					double jthActualValue = (yMatrix.getEntry(j, 0)==1)? 1: -1;	//Binarize the output
					double ithAlpha_Old = alpha[i];
					double jthAlpha_Old = alpha[j];

					double L, H = 0.0;
					//Use L_1 and H_1 if ithActualValue != jthActualValue
					if(ithActualValue != jthActualValue)	{
						L = Math.max(0, jthAlpha_Old- ithAlpha_Old);
						H = Math.min(C, C+ jthAlpha_Old - ithAlpha_Old);
					}
					
					//Use L_2 and H_2 if ithActualValue == jthActualValue					
					else{
						L = Math.max(0, jthAlpha_Old+ ithAlpha_Old - C);
						H = Math.min(C, jthAlpha_Old + ithAlpha_Old);	
					}
					
					if (L==H)
						continue;
					
					RealVector ithVector = xMatrix.getRowVector(i);
					RealVector jthVector = xMatrix.getRowVector(j);
					
					double ETA = 2*ithVector.dotProduct(jthVector) 
							- ithVector.dotProduct(ithVector) 
							- jthVector.dotProduct(jthVector);

					if(ETA >= 0)
						continue;
					
					double jthAlpha_New = jthAlpha_Old 
							- (jthActualValue/ETA)
							*(ithPredictionError - jthPredictionError);
					
					double jthAlpha_Clipped = 0.0;
					if(jthAlpha_New > H)
						jthAlpha_Clipped = H;
					else if (jthAlpha_New >= L && jthAlpha_New <= H)
						jthAlpha_Clipped = jthAlpha_New;
					else if(jthAlpha_New < L)
						jthAlpha_Clipped = L;
					
					alpha[j] = jthAlpha_Clipped; 
					if(Math.abs(alpha[j] - jthAlpha_Old) < 0.00001)
						continue;
					
					alpha[i] = ithAlpha_Old + ithActualValue*jthActualValue*(jthAlpha_Old - alpha[j]);
					double b1 = b- ithPredictionError 
							- ithActualValue*(alpha[i] - ithAlpha_Old)*ithVector.dotProduct(jthVector)
							- jthActualValue*(alpha[j] - jthAlpha_Old)*ithVector.dotProduct(jthVector);
					
					double b2 = b- jthPredictionError 
							- ithActualValue*(alpha[i] - ithAlpha_Old)*ithVector.dotProduct(jthVector)
							- jthActualValue*(alpha[j] - jthAlpha_Old)*ithVector.dotProduct(jthVector);
					
					//Update b
					if(alpha[i] > 0 && alpha[i] < C)
						b = b1;
					else if(jthAlpha_New > 0 && jthAlpha_New < C)
						b = b2;
					else
						b = (b1+b2)/2;
					
					nrOfChangedAlphas ++;
				}
				System.out.println("ithPredictionError: "+ ithPredictionError);
			}
			
			if(nrOfChangedAlphas == 0)
				passes ++;
			else
				passes = 0;
		}
		
		
		System.out.println("Test Complete");
		
		//Find accuracy
		double accuracy = 0.0;
		int nrOfCorrectPredictions = 0;
		for(int i=0; i< totalNrOfInstances; i++)	{
			double ithActualValue = yMatrix.getEntry(i, 0);
			double prediction = getHypothesisAtIndex(alpha, yMatrix, xMatrix, b, i);
			if(prediction < 0)	
				prediction = 3;
			else
				prediction = 1;
			
			 if(ithActualValue == prediction)
				 nrOfCorrectPredictions++;
		}
		accuracy = 100* (double) nrOfCorrectPredictions/ (double)totalNrOfInstances;
		System.out.println("Accuracy: "+ accuracy);
	}
	
	protected static double getHypothesisAtIndex(double[] alpha, RealMatrix yMatrix, RealMatrix xMatrix, double b, int index)	{
		
		int nrOfInstances = yMatrix.getRowDimension();
		double hypothesisAtIndex = 0.0;
		RealVector indexedVector = xMatrix.getRowVector(index);
		
		double ithTerm = 0.0;
		for(int i=0; i< nrOfInstances; i++)	{
			RealVector ithVector = xMatrix.getRowVector(i);
			double ithActualValue = (yMatrix.getEntry(i, 0)==1) ? 1 : -1;	//Binarize the output
			double cosineSim = ithVector.cosine(indexedVector);
			
			ithTerm = ithActualValue*alpha[i]*cosineSim;
					
			hypothesisAtIndex += ithTerm;
		}
		hypothesisAtIndex += b;
		
		/*if(hypothesisAtIndex > 1 || hypothesisAtIndex < -1)
			System.out.println("Attention!");*/
		
		return hypothesisAtIndex;
	}
	
	protected static double getPredictionErrorAtIndex(double[] alpha, RealMatrix yMatrix, RealMatrix xMatrix, double b, int index)	{
		
		double ithActualValue = (yMatrix.getEntry(index, 0)==1)? 1 : -1;	//Binarize the output
		double ithHypothesis = getHypothesisAtIndex(alpha, yMatrix, xMatrix, b, index);
		
		double predictionError = ithHypothesis - ithActualValue;
		
		/*if(predictionError > 2 || predictionError < -2)
			System.out.println("Attention!");*/
		
		return predictionError;
	}

}
