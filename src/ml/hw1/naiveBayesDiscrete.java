package ml.hw1;

import java.io.IOException;

public class naiveBayesDiscrete {

//	protected static final String DATA_INPUT_DIR = "src/main/resources/MlHw1Data/data_20newsgroups.csv";
//	protected static final String LABELS_INPUT_DIR = "src/main/resources/MlHw1Data/labels_20newsgroups.csv";
//	protected static final int NROFSPLITS = 10;
//	protected static final String TRAIN_PERCENT = "5 10 15 20 25 30";
	
	public static void main(String[] args) throws IOException {
		
		Hw2Main.yMatrix = InputOutput.initLabelsMatrix(args[1]);
		int nrOfDocs = Hw2Main.yMatrix.getRowDimension(); 
		InputOutput.initializeWordsInDoc(nrOfDocs);
		Hw2Main.usefulData = InputOutput.initDataMatrix(args[0], Hw2Main.yMatrix);
		Hw2Main.nrOfSplits = Integer.parseInt(args[2]);
		
		//Make a new arguments string to pass to the main classes
		String[] arguments = new String[args.length + 1];
		
		//0 for Logistic Regression and 1 for Naive Bayes
		arguments[0] = "1";
		for(int i=1; i< args.length + 1; i++)	{
			arguments[i] = args[i-1];	
		}
		
		Hw2Main.main(arguments);
	}

}
