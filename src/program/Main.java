package program;

import java.io.IOException;
import java.util.Random;

import model.NBayes;

import corpus.Corpus;
import corpus.Vocabulary;

public class Main {
	
	/** user parameters **/
	String delimiter = "\\+";
	int numIter = 40;
	long seed = 1;
	String inFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/nbayes/combined.final.propprocessed.span.word_hmm";
	String vocabFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/nbayes/combined.final.propprocessed.span.word_hmm";
	
	//String inFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.nolabel.txt";
	//String inFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.txt";
	boolean containsLabel = false;
	int numClass = 100; //not used if containsLabel = true;
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
		c.readDecodeInstance(vocabFile, containsLabel);
		c.read(inFile, containsLabel); 
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
			model.decodeLabeled(inFile + ".decoded");
		}
		else {
			model.decode(inFile + ".decoded");
		}
	}
	
	public void train() throws IOException {
		Corpus c = new Corpus(delimiter);
		c.readVocab(vocabFile, containsLabel);
		c.read(inFile, containsLabel); 
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
			model.decodeLabeled(inFile + ".decoded");
		}
		else {
			model.decode(inFile + ".decoded");
		}
	}
	
	public void test() throws IOException {
		Corpus c = new Corpus(delimiter);
		c.readVocabFromVocabFile(NBayes.base + "/dictionary.txt");
		c.readDecodeInstance(vocabFile, containsLabel);
		NBayes model;
		if(containsLabel) {
			model = new NBayes(c, c.labelMap.size(), c.corpusVocab.vocabSize, containsLabel);
			model.load();
			model.decodeLabeled(inFile + ".decoded");
		} else {
			model = new NBayes(c, numClass, c.corpusVocab.vocabSize, containsLabel);
			model.load();
			model.decode(inFile + ".decoded");
		}
	}
}
