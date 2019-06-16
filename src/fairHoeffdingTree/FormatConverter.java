/*
 * func: convert csv file to arff or vice versa
 */
package fairHoeffdingTree;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;

import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

public class FormatConverter {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
//		// 1. csv to arff:
//		//input file
//		String csvInputFileName = "censusSmall.csv"; 
//		String output_arff_FileName= "censusSmall.arff";
//		
//		// load csv
//		CSVLoader csvLoader= new CSVLoader();
//		csvLoader.setSource(new File("./data/"+ csvInputFileName));
//		Instances inputData= csvLoader.getDataSet();
//		
//		//save arff
//		ArffSaver arffSaver= new ArffSaver();
//		arffSaver.setInstances(inputData);
//		arffSaver.setFile(new File("./data/"+ output_arff_FileName));
//		arffSaver.writeBatch();
		
		// 2. arff to csv: 
		 //read arff
//		String arffInputFileName= "censusSmall.arff";
//		String output_csv_FileName= "adult.data_10.ToArff.csv";
//		
//		ArffReader arffReader= new ArffReader(new FileReader("./data/"+ arffInputFileName));
//		Instances data = arffReader.getData();
//		
//		CSVSaver csvSaver= new CSVSaver();
//		csvSaver.setInstances(data);
//		csvSaver.setFile(new File("./data/"+output_csv_FileName));
		
//		for (int i= 0; i< data.numInstances(); i++) {
//			System.out.println(data.instance(i));
//		}
		
		// 3. convert into numeric representation
		//input file
		String csvInputFileName = "censusSmall.csv"; 
		//String output_arff_FileName= "censusSmallConverted.csv";
		
		// load csv
		CSVLoader csvLoader= new CSVLoader();
		csvLoader.setSource(new File("./data/"+ csvInputFileName));
		Instances inputData= csvLoader.getDataSet();
		
		
		String outputFileName = "censusSmallConverted.csv";
		BufferedWriter br = new BufferedWriter(new FileWriter(new File("./data/results/" + outputFileName)));
		//Instances internalRepresentation= new Instances(inputData, inputData.numInstances());
		
		for (int i = 0; i < inputData.numAttributes(); i++) {
			br.write(inputData.attribute(i).name()+",");
		}
		br.write("\n");
		
		double temp[]= new double[inputData.numAttributes()];
		//temp= inputData.instance(2).toDoubleArray();
		//inputData.numInstances()
		for (int i = 0; i < inputData.numInstances(); i++) {
			temp= inputData.instance(i).toDoubleArray();
			for (int j = 0; j < temp.length; j++) {
				br.write(temp[j]+",");
			}
			br.write("\n");
			
//			for (int j2 = 0; j2 < temp.length; j2++) {
//				System.out.print(temp[j2]+ ", ");
//			}
//			System.out.println();
		}
		
		br.close();
		
		
		//inputData.attributeToDoubleArray(2);
//		for (int i = 0; i < inputData.numInstances(); i++) {
//			for (int j = 0; j < inputData.numAttributes(); j++) {
//				inputData.instance(i).toDoubleArray();
//			}
//		}
		
	
		//System.out.println(a);
		
//		for (int i = 0; i < a.length; i++) {
//			System.out.print(a[i]);
//		}
//		
//		System.out.println();
		
		//System.out.println(inputData.instance(1).toDoubleArray());
//		System.out.println(inputData.instance(1));
//		System.out.println(inputData.instance(1).value(13));
//		System.out.println(inputData.instance(2));
//		System.out.println(inputData.instance(2).value(0));
		
	}

}
