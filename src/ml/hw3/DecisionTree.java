package ml.hw3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class DecisionTree {

	protected static RealMatrix train2DecisionTree(
			int sampleIndex, RealMatrix[] sampledXYMatrix, ArrayList<Integer> restrictedAttriIds, int nrOfLevels) {

		RealMatrix xMatrix = sampledXYMatrix[0];
		RealMatrix yMatrix = sampledXYMatrix[1];
		int nrOfInstances = yMatrix.getRowDimension();
		int nrOfAttributes = xMatrix.getColumnDimension();
		int selectedParentAttributeId = -1;
		double infoGainParent = 1;
		int[] numBranches = new int[nrOfLevels+1];
		numBranches[0] = 1;
		double parentAttriMin = -1.0;
//		HashMap<String, Integer> bestLabelToPredict = oldBestLabelToPredict;
		RealMatrix decisionMatrix = new Array2DRowRealMatrix(225,6);
		int decisionRowsFilled = 0;
		
		//Loop over the levels of the decision tree and at each level find the best feature acc. to IG
		for(int level =0; level< nrOfLevels; level++)	{

			numBranches[level+1] = -1;
			double overallBestIg = -1;
			int bestAttriIndex = -1;
			double bestAttriMin = -1;
			
			for(int branch = 0; branch< numBranches[level]; branch++)	{
			
				overallBestIg = -1;
				bestAttriIndex = -1;
				bestAttriMin = -1;
				List<Integer> bestLabelsToPredict = new ArrayList<Integer>();
				
				//Loop over all attributes searching for the best one
				for(int col =0; col< nrOfAttributes; col++)	{
					if(col == selectedParentAttributeId || restrictedAttriIds.contains(col))
						continue;
					
					double[] attriCol = xMatrix.getColumn(col);
					Arrays.sort(attriCol);
					double attriMin = attriCol[0];
					double attriMax = attriCol[attriCol.length-1];
					int nrOfSplits = (int) Math.floor(attriMax- attriMin) + 1;
					double[] entropy = new double[nrOfSplits];
					double[] nrOfChildrenAtNode = new double[nrOfSplits];
					double totalChildrenCount = 0; 
					double weightedEntropy = 0;
					double infoGain = -1;
					List<Integer> labelsToPredict = new ArrayList<Integer>();
					int labelToPredict = -1;
					
					int split= 0;
					
					//Splits
					for(double i=attriMin; i<= attriMax; i++, split++)	{
						double origClass_C0_Count = 0.0;		
						double origClass_C1_Count = 0.0;
						double class_C0_Count = 0.0;		
						double class_C1_Count = 0.0;	
						labelToPredict = -1;
						
						if(level == 0)	{
							for(int id = 0; id< nrOfInstances; id++)	{
								double attriValue = xMatrix.getEntry(id, col);
								if(attriValue == i)	{
									if(yMatrix.getEntry(id,0) == -1)	
										class_C0_Count++;
									else
										class_C1_Count++;
								}
								else	{
									continue;
								}
							}	
						}
						else	{
							for(int id = 0; id< nrOfInstances; id++)	{
								//If parentAttribute value does not correspond to this branch, continue
								if(xMatrix.getEntry(id, selectedParentAttributeId) != parentAttriMin + branch)	{
									continue;
								}
								
								double attriValue = xMatrix.getEntry(id, col);
								//If this attribute value corresponds to this split value
								if(attriValue == i)	{
									if(yMatrix.getEntry(id,0) == -1)	{
										origClass_C0_Count++;
										class_C0_Count++;
									}
										
									else	{
										origClass_C1_Count++;
										class_C1_Count++;
									}
										
								}
								else	{
									if(yMatrix.getEntry(id,0) == -1)	
										origClass_C0_Count++;
									else
										origClass_C1_Count++;
									continue;
								}
//								System.out.println("ClassC0Count: "+ class_C0_Count+ " ClassC1Count: "+ class_C1_Count);
//								System.out.println("Orig.ClassC0Count: "+ origClass_C0_Count+ " Orig.ClassC1Count: "+ origClass_C1_Count);
							}
						}
						
						nrOfChildrenAtNode[split] = class_C0_Count+ class_C1_Count;
						if (nrOfChildrenAtNode[split] != 0)	{
							if(class_C0_Count != 0)
								entropy[split] = -(class_C0_Count/nrOfChildrenAtNode[split])*Math.log(class_C0_Count/nrOfChildrenAtNode[split])/Math.log(2);
							else
								entropy[split] = 0;
							
							if(class_C1_Count != 0)
								entropy[split] = entropy[split] -(class_C1_Count/nrOfChildrenAtNode[split])*Math.log(class_C1_Count/nrOfChildrenAtNode[split])/Math.log(2);	
						}
						 
						totalChildrenCount += nrOfChildrenAtNode[split]; 
						weightedEntropy += nrOfChildrenAtNode[split]*entropy[split];
						
						if(class_C0_Count == 0 & class_C1_Count == 0)	{
							if(origClass_C1_Count > origClass_C0_Count)	{
								labelToPredict = 1;
							}
							else
								labelToPredict = 0;
						}
						else	{
							if(class_C1_Count > class_C0_Count)	{
								labelToPredict = 1;
							}
							else
								labelToPredict = 0;
						}
						
						labelsToPredict.add(labelToPredict);
//						bestLabelToPredict.put(sampleIndex+ "\t"+ selectedParentAttributeId+ "\t"+ (parentAttriMin + branch)+ "\t"+ col+ "\t"+i, labelToPredict);
//						System.out.println("For "+ sampleIndex+ "\t"+ selectedParentAttributeId+ "\t"+ (parentAttriMin + branch)+ "\t"+ col+ "\t"+ i+ "\tLabel: "+ labelToPredict);
//						System.out.println("Testing...");	
					}
					
					weightedEntropy = weightedEntropy/totalChildrenCount;
					infoGain = infoGainParent- weightedEntropy; 
					
//					System.out.println("Level: "+ level+ " Branch: "+ branch+ " Att. Index: "+ col+ " NrSplits: "+ split+ " Current rel. info gain: "+ infoGain + "\tPrevious best IG: "+ overallBestIg);
					if(infoGain > overallBestIg)	{
						overallBestIg = infoGain;
						bestAttriIndex = col;
						numBranches[level+1] = nrOfSplits;
						bestAttriMin = attriMin;
						bestLabelsToPredict = labelsToPredict;
					}
					
//					When total children count = 0
					if(bestLabelsToPredict.size() == 0)	{
						bestLabelsToPredict = labelsToPredict;
					}
					System.out.print("");
				}
				
				
				//Update the decision Matrix
				if(level == 1)	{
					int priorRowsFilled = decisionRowsFilled; 
					for(int i=priorRowsFilled; i< priorRowsFilled+ numBranches[level+1]; i++)	{
						decisionMatrix.setEntry(i,0,sampleIndex+1);
						decisionMatrix.setEntry(i,1,selectedParentAttributeId);
						decisionMatrix.setEntry(i,2,parentAttriMin + branch);
						decisionMatrix.setEntry(i,3,bestAttriIndex);
						decisionMatrix.setEntry(i,4,bestAttriMin+ i-priorRowsFilled);
						decisionMatrix.setEntry(i,5,bestLabelsToPredict.get(i-priorRowsFilled));
						
						decisionRowsFilled++;
					}
				}
				
//				System.out.println("Testing...");	
			}
			selectedParentAttributeId = bestAttriIndex;
			parentAttriMin = bestAttriMin;
			infoGainParent = overallBestIg;
//			newClassifierParams[sampleIndex][0] = bestAttriIndex;
			
//			System.out.println("Overall best IG: "+ overallBestIg+ "\tfinalSplitValue\t"+ finalSplitValue+ "\tbestAttriIndex\t"+ bestAttriIndex);
//			System.out.println("Testing...");	
			
		}

		return decisionMatrix;
	}

	protected static RealMatrix train1DecisionTree(int sampleIndex,
			RealMatrix[] sampledXYMatrix,
			ArrayList<Integer> restrictedAttriIds, int nrOfLevels) {

		RealMatrix xMatrix = sampledXYMatrix[0];
		RealMatrix yMatrix = sampledXYMatrix[1];
		int nrOfInstances = yMatrix.getRowDimension();
		int nrOfAttributes = xMatrix.getColumnDimension();
		RealMatrix decisionMatrix = new Array2DRowRealMatrix(50,6);
				
				//Loop over all attributes searching for the best one
				for(int col =0; col< nrOfAttributes; col++)	{
					if(restrictedAttriIds.contains(col))
						continue;
					
					int attributeId = col;
					double[] attriCol = xMatrix.getColumn(col);
					Arrays.sort(attriCol);
					double attriMin = attriCol[0];
					double attriMax = attriCol[attriCol.length-1];
					List<Integer> labelsToPredict = new ArrayList<Integer>();
					int labelToPredict = -1;
					
					//Splits
					for(double splitIndex=attriMin; splitIndex<= attriMax; splitIndex++)	{
						double origClass_C0_Count = 0.0;		
						double origClass_C1_Count = 0.0;
						double class_C0_Count = 0.0;		
						double class_C1_Count = 0.0;	
						labelToPredict = -1;
						
						for(int id = 0; id< nrOfInstances; id++)	{
							double attriValue = xMatrix.getEntry(id, col);
							if(attriValue == splitIndex)	{
								if(yMatrix.getEntry(id,0) == -1)	{
									origClass_C0_Count++;
									class_C0_Count++;
								}
									
								else	{
									origClass_C1_Count++;
									class_C1_Count++;
								}
							}
							else	{
								if(yMatrix.getEntry(id,0) == -1)	{
									origClass_C0_Count++;
								}
									
								else	{
									origClass_C1_Count++;
								}
							}
						}	
						
						if(class_C0_Count == 0 & class_C1_Count == 0)	{
							if(origClass_C1_Count > origClass_C0_Count)	{
								labelToPredict = 1;
							}
							else
								labelToPredict = 0;
						}
						else	{
							if(class_C1_Count > class_C0_Count)	{
								labelToPredict = 1;
							}
							else
								labelToPredict = 0;
						}
						
						labelsToPredict.add(labelToPredict);
//						System.out.println("Testing...");	
					}
					
					System.out.print("");
					//Update the decision Matrix
					for(int splitIndex=(int) attriMin; splitIndex<= attriMax; splitIndex++)	{
						int index = splitIndex - (int)attriMin;
						decisionMatrix.setEntry(index,0,sampleIndex+1);
						decisionMatrix.setEntry(index,1,attributeId);
						decisionMatrix.setEntry(index,2,splitIndex);
						decisionMatrix.setEntry(index,3,-1);
						decisionMatrix.setEntry(index,4,-1);
						decisionMatrix.setEntry(index,5,labelsToPredict.get(splitIndex-(int)attriMin));
					}
				}
//				System.out.println("Testing...");	
				return decisionMatrix;
	}
	
}
