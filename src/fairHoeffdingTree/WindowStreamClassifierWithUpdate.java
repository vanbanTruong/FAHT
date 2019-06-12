/*
 * Window based stream classifier processes window based instances in contrast to predict and update instance one by one.
 * hoeffding trees in the classifier window will also GET updated with instances in the current window
 * 
 * 
 * ***The hoeffding tree here is the fair splitting criteria embedded based on weka.***
 */
package fairHoeffdingTree;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class WindowStreamClassifierWithUpdate {
	
	private static String saName = "sex"; // sensitive attribute name
	private static String saValue = "Female"; // sensitive attribute value
	private static int saIndex; // index of sensitive attribute 

	private static int indexOfDeprived = -1; // sensitive attribute: female
	private static int indexOfUndeprived = -1; // sensitive attribute: male
	private static int indexOfDenied = -1; // class label: income <=50k
	private static int indexOfGranted = -1; // class label: income > 50k

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		// Import data
		String inputFileName = "adult.data.csv";
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("./data/" + inputFileName));
		loader.setNoHeaderRowPresent(false);
		
		Instances stream = loader.getDataSet();
		stream.setClassIndex(stream.numAttributes() - 1);
		System.out.println("load data successfully!");

		// Initialize experiment parameters
		int windowSize = 0; // incrementally controlled by the for loop below
		int windowSizeOfClassifer = 10; // how many classifiers for ensemble
		
		saIndex= stream.attribute(saName).index();
		indexOfDeprived = stream.attribute(saName).indexOfValue(saValue); // M:0 F:1
		indexOfUndeprived = stream.attribute(saName).indexOfValue("Male");
		indexOfDenied = stream.classAttribute().indexOfValue("<=50K"); // <=50K: 0, >50K: 1
		indexOfGranted = stream.classAttribute().indexOfValue(">50K");
		

		String outputFileName = "fair-updated-"+inputFileName + "_" + windowSizeOfClassifer + ".csv";
		BufferedWriter br = new BufferedWriter(new FileWriter(new File("./data/results/" + outputFileName)));
		br.write("windowSize, accuracy, discrimination, fairaccuracy\n");

		WindowStreamClassifierWithUpdate wsc = new WindowStreamClassifierWithUpdate();
		int counteri = 0;
		int incrementSize = 100;
		int maxWindowSize = 1000;
		int iterationTime = maxWindowSize / incrementSize;

		// incrementing window size
		for (windowSize = 100; windowSize <= maxWindowSize; windowSize += incrementSize) {
			wsc.trainClassifier(stream, br, windowSize, windowSizeOfClassifer);
			System.out.println("this is the " + (++counteri) + "/" + iterationTime + "th iteration");
		}

		br.close();
		System.out.println("task finished!");

	}

	protected void trainClassifier(Instances stream, BufferedWriter br, int slideWindowSize,
			int slideWindowSizeOfClassifier) throws Exception {

		int windowSize = slideWindowSize; // size of current window
		int windowSizeOfClassifier = slideWindowSizeOfClassifier; // size of window used to store classifier
		int numOfInstanceInWindow = 0; // count the # of instance in the window
		int numOfClassifierCounter = 0; // count the # of classifiers in classifier window
		int numOfCurrentWindow = 0; // store the current window number

		int numOfWindow = (stream.numInstances() / windowSize) + 1; // estimated number of window
		// Record each sliding window's classification statistics
		// first elements of those arrays are 0 as numOfWindow starts from 1
		double[] accuracy = new double[numOfWindow];
		double[] discrimination = new double[numOfWindow];

		// Two windows
		Instances currentWindow = new Instances(stream, windowSize);
		HoeffdingTree[] classifierList = new HoeffdingTree[windowSizeOfClassifier];

		// train and classify current window
		for (int i = 0; i < stream.numInstances(); i++) {
			if (numOfInstanceInWindow < windowSize) {
				Instance intance = stream.instance(i);
				currentWindow.add(intance);
				numOfInstanceInWindow++;
			} else {
				numOfCurrentWindow++; // starts from 1

				if (numOfClassifierCounter < 1) { // 1. When the window of classifiers contains no classifier.
					HoeffdingTree classifer = new HoeffdingTree();
					currentWindow.setClassIndex(currentWindow.numAttributes() - 1); // set class attribute index
					classifer.buildClassifier(currentWindow);
					// add classifier
					classifierList[0] = classifer;
					numOfClassifierCounter++;

					// Record this sliding window's classification statistics
					double results[] = performanceStatistics(currentWindow, numOfClassifierCounter, classifierList);
					accuracy[numOfCurrentWindow] = results[0];
					discrimination[numOfCurrentWindow] = results[1];

				} else {
					// 2. There are classifiers in the classifiers window.
					// Classify current sliding window using previous trained classifiers.
					double results[] = performanceStatistics(currentWindow, numOfClassifierCounter, classifierList);
					accuracy[numOfCurrentWindow] = results[0];
					discrimination[numOfCurrentWindow] = results[1];

					// train a new classifier on current window
					HoeffdingTree newClassifer = new HoeffdingTree();
					newClassifer.buildClassifier(currentWindow);
					
					// add the newly trained
					if (numOfClassifierCounter < windowSizeOfClassifier) {
						classifierList[numOfClassifierCounter] = newClassifer;
						numOfClassifierCounter++;
					} else {
						for (int j = 0; j < (windowSizeOfClassifier - 1); j++) {
							classifierList[j] = classifierList[j + 1];
						}
						classifierList[windowSizeOfClassifier - 1] = newClassifer;
					}
					
					/**
					 * update other classifiers in the classifier window
					 * -2: the last one is the newly trained and added
					 */
					for (int j = 0; j <= (numOfClassifierCounter - 2); j++) {
						for (int j1 = 0; j1 < currentWindow.numInstances(); j1++) {
							classifierList[j].updateClassifier(currentWindow.instance(j1));
						}	 
					}
						
				}

				currentWindow.clear(); // empty current sliding window
				numOfInstanceInWindow = 0;

			}
		} // end of for loop

		// Get overall statistics
		double overallAccuray = 0;
		double overallDiscrimination = 0;
		double overallFairaccuracy = 0;

		double tempAcc = 0, tempDis = 0;
		// accuracy[numOfCurrentWindow]: numOfCurrentWindow starts from 1 and ends at
		// numOfCurrentWindow
		for (int j = 1; j <= numOfCurrentWindow; j++) {
			tempAcc += accuracy[j];
			tempDis += discrimination[j];
		}

		overallAccuray = tempAcc / numOfCurrentWindow;
		overallDiscrimination = tempDis / numOfCurrentWindow;
		overallFairaccuracy = overallAccuray - overallDiscrimination;

		// write results
		br.write(windowSize + "," + overallAccuray + "," + overallDiscrimination + "," + overallFairaccuracy + "\n");

	}

	protected double[] performanceStatistics(Instances data, int numOfClassifierCounter, HoeffdingTree[] classifierList)
			throws Exception {

		int windowSize = data.numInstances();

		// predicts class label
		int[] predictedClassLabel = new int[windowSize];
		for (int j = 0; j < windowSize; j++) {
			double grantedProb = 0;
			double deniedProb = 0;
			for (int j2 = 0; j2 < numOfClassifierCounter; j2++) {
				grantedProb += classifierList[j2].distributionForInstance(data.instance(j))[indexOfGranted];
				deniedProb += classifierList[j2].distributionForInstance(data.instance(j))[indexOfDenied];
			}

			if (grantedProb >= deniedProb) {
				predictedClassLabel[j] = indexOfGranted;
			} else {
				predictedClassLabel[j] = indexOfDenied;
			}
		}

		// Calculate accuracy and discrimination of current window
		int corrPredictedCount = 0; // The number of correctly classified instances
		int undeprivedPredictedCount = 0; // undeprived attribute being classified as granted
		int deprivedPredictedCount = 0; // deprived attribute being classified as granted, not necessary true class
										// label is granted

		// the number of instances with undeprived and deprived attribute respectively
		int numOfUndeprived = data.attributeStats(saIndex).nominalCounts[indexOfUndeprived];
		int numOfDeprived = data.attributeStats(saIndex).nominalCounts[indexOfDeprived];

		for (int j = 0; j < windowSize; j++) {
			int actualClassLabel = (int) data.instance(j).classValue();

			if (predictedClassLabel[j] == actualClassLabel) {
				corrPredictedCount++;
			}

			if ((data.instance(j).value(saIndex) == indexOfUndeprived) && (predictedClassLabel[j] == indexOfGranted)) {
				undeprivedPredictedCount++;
			}

			if ((data.instance(j).value(saIndex) == indexOfDeprived) && (predictedClassLabel[j] == indexOfGranted)) {
				deprivedPredictedCount++;
			}
		}

		double accuracy = (double) corrPredictedCount / windowSize;
		double discrimination = ((double) undeprivedPredictedCount / numOfUndeprived)
				- ((double) deprivedPredictedCount / numOfDeprived);
		double statistics[] = {accuracy, discrimination};

		return statistics;
	}

}
