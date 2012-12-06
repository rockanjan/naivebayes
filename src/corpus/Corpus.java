package corpus;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Corpus {
	//public static String delimiter = "\\+";
	public String delimiter;
	public InstanceList trainInstanceList = new InstanceList();
	
	public InstanceList decodeInstanceList = new InstanceList();
	
	//for label
	public int labelIdCount;
	public Map<String, Integer> labelMap;
	public ArrayList<String> labelIdToString;
	
	public Vocabulary corpusVocab; 
	
	public Corpus(String delimiter) {
		this.delimiter = delimiter;
	}
	
	public void read(String inFile, boolean containsLabel) throws IOException {
		Vocabulary trainVocab = new Vocabulary();
		trainVocab.readVocabFromFile(this, inFile, containsLabel);
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
		System.out.println("Train vocab size : " + trainVocab.vocabSize);
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
		System.out.println("Decode Instances: " + decodeInstanceList.size());
		System.out.println("Decode token count: " + totalWords);
		br.close();
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
		c.read(inFile, false);
	}
}
