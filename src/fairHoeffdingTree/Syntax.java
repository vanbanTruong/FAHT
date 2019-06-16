package fairHoeffdingTree;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
import moa.streams.generators.RandomRBFGenerator;

public class Syntax {
	public IntOption gracePeriodOption = new
			IntOption("gracePeriod", 'g',"The number of instances a leaf should observe between split attempts.",
					200, 0, Integer.MAX_VALUE);
	
	public static void main(String[] args) {
		Classifier learner= new HoeffdingTree();
		RandomRBFGenerator stream=new RandomRBFGenerator();   
		learner.prepareForUse();
		
		HoeffdingTree learner1= new HoeffdingTree();
		learner1.gracePeriodOption.setValue(100);
		
		Syntax syntax= new Syntax();
		syntax.gracePeriodOption.setValue(200);
		
		
		


		
		//Use instance: 在moa里面是以InstanceExample的形式存储的
		//1.
		Instance trainInst1= stream.nextInstance().getData(); //getData(): InstanceExample-> Instance
		//2.
		InstanceExample trainInst2= stream.nextInstance();
		
		learner.correctlyClassifies(trainInst1);
		learner.trainOnInstance(trainInst2);
	}
	
	
	
}
