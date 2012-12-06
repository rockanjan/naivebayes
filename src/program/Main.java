package program;

import java.io.IOException;
import java.util.Random;

import model.NBayes;

import corpus.Corpus;
import corpus.Vocabulary;

public class Main {

	/**
	 * @param args
	 * @throws IOException 
	 */
	
	public static void main(String[] args) throws IOException {
		/** user parameters **/
		String delimiter = ",";
		int numIter = 50;
		long seed = 1;
		//String inFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/nbayes/combined.final.propprocessed.span";
		String inFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.nolabel.txt";
		//String inFile = "/home/anjan/workspace/naivebayes/data/weather.nominal.txt";
		boolean containsLabel = false;
		int numClass = 2; //not used if containsLabel = true;
		/** user parameters end **/
		
		Corpus c = new Corpus(delimiter);
		c.read(inFile, containsLabel); //also reads the vocabulary
		Random r = new Random(seed);
		NBayes model;
		if(containsLabel) {
			model = new NBayes(c, c.labelMap.size(), Vocabulary.vocabSize, containsLabel);
			model.initializeSupervised();
			//model.train(1);
		} else {
			model = new NBayes(c, numClass, Vocabulary.vocabSize, containsLabel);
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
}
