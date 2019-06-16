/*
 * func: discrimination aware classification in data streams employing hoeffding tree.
 */
package fairHoeffdingTree;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.lazy.kNN;
import moa.classifiers.meta.OzaBag;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.ArffFileStream;


public class FairStreamClassifier {
	
	private static int windowSize= 1000;
	private static String saName= "sex"; //sensitive attribute name 
	private static String saVale= "Female"; //sensitive attribute value 
	private static int saIndex; //sensitive attribute index of the stream
	//private static String className= "class"; //class name

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		//Import data
		String inputFile= "test.arff";
		System.out.println("load data successfully!");
		String outputFile= inputFile+ "_"+ windowSize+ ".csv";
		//BufferedWriter br= new BufferedWriter(new FileWriter(new File("./data/results/"+outputFile)));
		//br.write("windowNum, accuracy, discrimination, fairaccuracy,"+ "overallAccuracy, overallDiscrimination, overallFairaccuracy\n");
		
		ArffFileStream stream= new ArffFileStream("./data/"+ inputFile, 14);
		stream.prepareForUse();
		
		FairStreamClassifier fairStreamClassifier= new FairStreamClassifier();
		fairStreamClassifier.trainClassifier(stream);
		//br.close();
		
		System.out.println("task finished!");
		
	}
	
	protected void trainClassifier(ArffFileStream stream) {
		
		//find sensitive attribute index of the stream
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			if (stream.getHeader().attribute(i).name().equals(saName)) {
				saIndex = i;
				break;
			}
		}
		
		//counts
		int numCorrectClassified= 0;
		int numUndeprived= 0, numDeprived= 0;
		int undeprivedPredictedCount= 0, deprivedPredictedCount= 0;
		int numOfInstances= 0;
		
		int indexOfUndeprived= stream.getHeader().attribute(saIndex).indexOfValue("Male"); //M: 1 F: 0
		int indexOfDeprived= stream.getHeader().attribute(saIndex).indexOfValue(saVale);
		int indexOfGranted= 1;
		int indexOfDenied= 0;
		
		int trueLabel= -1;
		int predictedLabel= -1;
		boolean correctClassification= false;
		
		// classifier
		//Classifier hTree = new OzaBag(); //84.52% 22.19%
		//Classifier hTree = new kNN(); //81.90%  23.80% discrimination
		HoeffdingTree hTree= new HoeffdingTree(); //83.95% accuracy 24.17% dis
		hTree.setModelContext(stream.getHeader());
		hTree.splitConfidenceOption.setValue(0.01); //0.0000001
		//System.out.println("splitConfidenceOption:"+ hTree.splitConfidenceOption.getValue());
		//hTree.gracePeriodOption.setValue(1); //47,985?
		hTree.prepareForUse();
		//undeprivedCount:32650, deprivedCount:16192 48842
		while (stream.hasMoreInstances()) {
			numOfInstances++;
			Instance inst= stream.nextInstance().getData();
			trueLabel= (int) inst.classValue();
			correctClassification= hTree.correctlyClassifies(inst);
	
			if (trueLabel== indexOfGranted) {
				predictedLabel= correctClassification? indexOfGranted:indexOfDenied;
			} else {
				predictedLabel= correctClassification? indexOfDenied:indexOfGranted;
			}
			
			//calculate accuracy
			if (correctClassification) {
				numCorrectClassified++;
			}
			
			//calculate discrimination
			if (inst.value(saIndex)== indexOfUndeprived) {
				numUndeprived++;
				if (predictedLabel== indexOfGranted) {
					undeprivedPredictedCount++;
				}
			}
			
			if (inst.value(saIndex)== indexOfDeprived) {
				numDeprived++;
				if (predictedLabel== indexOfGranted) {
					deprivedPredictedCount++;
				}
				
			}
			
			hTree.trainOnInstance(inst);		
		}
		
		System.out.println("undeprivedCount:"+numUndeprived+ ", deprivedCount:"+ numDeprived);
		System.out.println(numUndeprived+numDeprived);
		
		double accuracy= 100*(double)numCorrectClassified/numOfInstances;
		double discrimination= 100*((double)undeprivedPredictedCount/numUndeprived)-((double)deprivedPredictedCount/numDeprived);
		
		System.out.println(hTree);
		System.out.println(numOfInstances + " instances processed with " + accuracy + "% accuracy");
		System.out.println(numOfInstances + " instances processed with " + discrimination + "% discrimination");
	}

}
