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
		int numIter = 40;
		long seed = 1;
		String inFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/nbayes/combined.final.propprocessed.span";
		int numClass = 100;
		Corpus.read(inFile); //also reads the vocabulary
		NBayes model = new NBayes(numClass, Vocabulary.vocabSize);
		
		Random r = new Random(seed);
		model.initializeRandom(r);
		model.train(numIter);
		model.save();
		model.decode();
	}
}
