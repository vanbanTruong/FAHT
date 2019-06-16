package fairHoeffdingTree;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.lazy.kNN;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.ArffFileStream;
import moa.streams.generators.RandomRBFGenerator;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// //Import data
		// String inputFile= "./data/"+"censusSmall.arff";
		// String outFile="";
		// System.out.println("load data successfully!");
		//
		// ArffFileStream stream= new ArffFileStream(inputFile, 12);
		// stream.prepareForUse();
		//
		// //System.out.println(stream.nextInstance().getData());
		// System.out.println(stream.getHeader());

		// //*****HoeffdingTree()
		// RandomRBFGenerator stream = new RandomRBFGenerator();
		// stream.prepareForUse();
		// Classifier learner = new HoeffdingTree();
		// learner.setModelContext(stream.getHeader());
		// learner.prepareForUse();
		// int numInstances=100000;
		// int numberSamplesCorrect=0;
		// int numberSamples=0;
		// boolean isTesting = true;
		// while(stream.hasMoreInstances() && numberSamples < numInstances){
		// Instance inst =stream.nextInstance().getData();
		// if(isTesting){
		// if(learner.correctlyClassifies(inst)){
		// numberSamplesCorrect++;
		// }
		// }
		// numberSamples++;
		// learner.trainOnInstance(inst);
		// }
		//
		// double accuracy = 100.0*(double)numberSamplesCorrect/(double)numberSamples;
		// System.out.println(numberSamples+" instances processed with "+accuracy+"%
		// accuracy");
		//
		// //*****End of HoeffdingTree()

		// *****HoeffdingTree()
		// Cross validation: stream.getHeader().trainCV(arg0, arg1)
		String inputFile = "censusSmall.arff";
		ArffFileStream stream= new ArffFileStream("./data/"+ inputFile, 13);
		stream.prepareForUse();
		Classifier learner = new HoeffdingTree(); //91.63%
		//Classifier learner = new kNN(); //91.65%
		learner.setModelContext(stream.getHeader());
		learner.prepareForUse();
		
		int numInstances= 10;
		int numberSamplesCorrect = 0;
		int numberSamples = 0;
		boolean isTesting = true;
		while (stream.hasMoreInstances() && numberSamples< numInstances) {
			Instance inst = stream.nextInstance().getData();
//			for (int i = 0; i < learner.getVotesForInstance(inst).length; i++) {
//				System.out.println(learner.getVotesForInstance(inst)[i]);
//			}
			
			//System.out.println("prediction:"+ learner.getPredictionForInstance(inst));
			
			//System.out.println("vote array length: "+learner.getVotesForInstance(inst).length);
			if (isTesting) {
				if (learner.correctlyClassifies(inst)) {
					numberSamplesCorrect++;
				}
			}
			numberSamples++;
			learner.trainOnInstance(inst);
			
		}

		double accuracy = 100.0 * (double) numberSamplesCorrect / (double) numberSamples;
		System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy");
		//System.out.println(learner);
		//System.out.println(stream.getHeader());

		// *****End of HoeffdingTree()
		
		//System.out.println(stream.getHeader());
		//Attribute value
		//Instance inst1= stream.nextInstance().getData();
		//Instance inst2= stream.nextInstance().getData();

		//System.out.println("~"+inst1);
		//System.out.println("~"+inst1.value(inst1.numAttributes()-1)+":");
		//System.out.println("~"+inst2);
		//System.out.println("~"+inst2.value(inst1.numAttributes()-1)+":");
		System.out.println(102/50+1);
	}

}
