# FAHT
Implementation of FAHT (IJCAI 19), a fair classifier for online stream based decision-making. Detailded information about FAHT is provided in [FAHT](https://www.ijcai.org/proceedings/2019/0205.pdf).  

## Instructions
1. Clone this repository.
2. Download the datasets as described in the Experiment/Data folder of this repository to the root folder.
3. Run the code with Weka > 3.9.
  ### In Experiment folder: InstanceStreamClassifier.java and WindowStreamClassifier.java evaluate the landmark window model and     sliding window model, respectively.
  ### The FAHT folder contains the source code.
  
## Citation
@inproceedings{zhang2019faht,  
  title={FAHT: an adaptive fairness-aware decision tree classifier},  
  author={Zhang, Wenbin and Ntoutsi, Eirini},  
  booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence},  
  pages={1480--1486},  
  year={2019},  
  organization={AAAI Press}  
}
