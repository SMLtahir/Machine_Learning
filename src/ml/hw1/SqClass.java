package ml.hw1;

import java.io.IOException;
import java.util.List;

public class SqClass {

//	protected static final String SPAM_INPUT_DIR = "src/main/resources/MlHw1Data/spam.csv";
//	protected static final String MNIST_INPUT_DIR = "src/main/resources/MlHw1Data/MNIST-1378.csv";
	public static void main(String[] args) throws IOException {

		//Initialize data instances
		List<String> instances = new InputOutput().initInstances(args[0]);
		
		if(Hw1Main.givenClassLabels.length == 2)
			Hw1_3_3_2Class.main(args);
		else
			Hw1_3_3_4Class.main(args);				

	}

}
