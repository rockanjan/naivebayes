package program;

import java.io.IOException;
import java.util.Random;

import model.NBayes;

import corpus.Corpus;
import corpus.Vocabulary;

public class Main {
	
	/** user parameters **/
	String delimiter = "\\+";
	int numIter = 50;
	long seed = 1;
	
	//String trainFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/paper/nbayes/combined.nbayes.after";
	//String vocabFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/paper/nbayes/combined.nbayes.after";
	//String testFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/paper/nbayes/combined.nbayes.after";
	/*
	String trainFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.txt";
	String vocabFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.txt";
	String testFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.txt";
	*/
	
	String trainFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.nolabel.txt";
	String vocabFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.nolabel.txt";
	String testFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.nolabel.txt";
	
	boolean containsLabel = false;
	int numClass = 4; //not used if containsLabel = true;
	/** user parameters end **/
	
	public static void main(String[] args) throws IOException {
		Main main = new Main();
		main.train();
		//main.continueTrain();
		//main.test();
	}
	
	public void continueTrain() throws IOException {
		Corpus c = new Corpus(delimiter);
		c.readVocabFromVocabFile(NBayes.base + "/dictionary.txt");
		c.readTrain(trainFile, containsLabel);
		c.readTest(testFile, containsLabel);
		NBayes model;
		if(containsLabel) {
			model = new NBayes(c, c.labelMap.size(), c.corpusVocab.vocabSize, containsLabel);
			model.load();
			model.train(numIter);
		} else {
			model = new NBayes(c, numClass, c.corpusVocab.vocabSize, containsLabel);
			model.load();
			model.train(numIter);
		}
		model.save();
		if(containsLabel) {
			model.decodeLabeled(testFile + ".decoded");
		}
		else {
			model.decode(testFile + ".decoded");
		}
	}
	
	public void train() throws IOException {
		Corpus c = new Corpus(delimiter);
		c.readVocab(vocabFile, containsLabel);
		c.readTrain(trainFile, containsLabel);
		c.readTest(testFile, containsLabel); 
		Random r = new Random(seed);
		NBayes model;
		if(containsLabel) {
			model = new NBayes(c, c.labelMap.size(), c.corpusVocab.vocabSize, containsLabel);
			model.initializeSupervised();
			//model.train(1);
		} else {
			model = new NBayes(c, numClass, c.corpusVocab.vocabSize, containsLabel);
			model.initializeRandom(r);
			model.train(numIter);
		}
		model.save();
		if(containsLabel) {
			model.decodeLabeled(testFile + ".decoded");
		}
		else {
			model.decode(testFile + ".decoded");
		}
	}
	
	public void test() throws IOException {
		Corpus c = new Corpus(delimiter);
		if(containsLabel) {
			c.readLabels(NBayes.base + "/label.txt");
		}
		c.readVocabFromVocabFile(NBayes.base + "/dictionary.txt");
		c.readTest(testFile, containsLabel);
		NBayes model;
		if(containsLabel) {
			model = new NBayes(c, c.labelMap.size(), c.corpusVocab.vocabSize, containsLabel);
			model.load();
			model.decodeLabeledVector(testFile + ".decoded");
		} else {
			model = new NBayes(c, numClass, c.corpusVocab.vocabSize, containsLabel);
			model.load();
			model.decode(testFile + ".decoded");
		}
	}
}
