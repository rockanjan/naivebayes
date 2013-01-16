package corpus;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Corpus {
	//public static String delimiter = "\\+";
	public String delimiter;
	public InstanceList trainInstanceList = new InstanceList();
	public InstanceList testInstanceList = new InstanceList();
	
	//for label
	public int labelIdCount;
	public Map<String, Integer> labelMap;
	public ArrayList<String> labelIdToString;
	
	public Vocabulary corpusVocab; 
	
	public Corpus(String delimiter) {
		this.delimiter = delimiter;
	}
	
	public void readTest(String inFile, boolean containsLabel) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(inFile));
		String line = null;
		int totalWords = 0;
		testInstanceList = new InstanceList();
		while( (line = br.readLine()) != null ) {
			line = line.trim();
			if(! line.isEmpty()) {
				Instance instance = new Instance(this, line, containsLabel);
				testInstanceList.add(instance);
				totalWords += instance.words.length;
			}
		}
		System.out.println("Test Instances: " + testInstanceList.size());
		System.out.println("Test token count: " + totalWords);
		br.close();
	}
	
	public void readTrain(String inFile, boolean containsLabel) throws IOException {
		//Vocabulary trainVocab = new Vocabulary();
		//trainVocab.readVocabFromFile(this, inFile, containsLabel);
		BufferedReader br = new BufferedReader(new FileReader(inFile));
		String line = null;
		int totalWords = 0;
		while( (line = br.readLine()) != null ) {
			line = line.trim();
			if(! line.isEmpty()) {
				Instance instance = new Instance(this, line, containsLabel);
				trainInstanceList.add(instance);
				totalWords += instance.words.length;
			}
		}
		//System.out.println("Train vocab size : " + trainVocab.vocabSize);
		System.out.println("Train Instances: " + trainInstanceList.size());
		System.out.println("Train token count: " + totalWords);
		br.close();
	}
	
	public void readVocab(String inFile, boolean containsLabel) throws IOException {
		corpusVocab = new Vocabulary();
		if(containsLabel) {
			labelIdCount = 0;
			labelMap = new HashMap<String, Integer>();
			labelIdToString = new ArrayList<String>();
		}
		corpusVocab.readVocabFromFile(this, inFile, containsLabel);
		/*
		BufferedReader br = new BufferedReader(new FileReader(inFile));
		String line = null;
		int totalWords = 0;
		while( (line = br.readLine()) != null ) {
			line = line.trim();
			if(! line.isEmpty()) {
				Instance instance = new Instance(this, line, containsLabel);
				decodeInstanceList.add(instance);
				totalWords += instance.words.length;
			}
		}
		System.out.println("Vocab Instances: " + decodeInstanceList.size());
		System.out.println("Vocab token count: " + totalWords);
		br.close();
		*/
	}
	
	public void readVocabFromVocabFile(String filename) {
		corpusVocab = new Vocabulary();
		corpusVocab.readVocabFromVocabFile(filename);
	}
	
	public void readLabels(String filename) throws IOException {
		System.out.println("\treading labels...");
		BufferedReader brLabel = new BufferedReader(new FileReader(filename));
		String	line = brLabel.readLine().trim();
		this.labelIdToString = new ArrayList<String>();
		this.labelMap = new HashMap<String, Integer>();
		while( (line = brLabel.readLine() ) != null) {
			line = line.trim();
			this.labelMap.put(line, this.labelIdToString.size());
			this.labelIdToString.add(line);
			
		}
		brLabel.close();
	}
	
	public int getLabelMap(String label) {
		if(labelMap.containsKey(label)) {
			return labelMap.get(label);
		} else {
			labelMap.put(label, labelIdCount);
			labelIdToString.add(label);
			return labelIdCount++;
		}
	}
	
	public static void main(String[] args) throws IOException {
		String inFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/nbayes/combined.final.propprocessed.span";
		Corpus c = new Corpus("\\+");
		c.readVocab(inFile, false);
	}
}
