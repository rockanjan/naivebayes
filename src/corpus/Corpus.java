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
	public InstanceList instanceList = new InstanceList();
	
	//for label
	public int labelIdCount;
	public Map<String, Integer> labelMap;
	public ArrayList<String> labelIdToString;
	
	public Corpus(String delimiter) {
		this.delimiter = delimiter;
	}
	
	public void read(String inFile, boolean containsLabel) throws IOException {
		if(containsLabel) {
			labelIdCount = 0;
			labelMap = new HashMap<String, Integer>();
			labelIdToString = new ArrayList<String>();
		}
		Vocabulary.readVocabFromFile(this, inFile, containsLabel);
		BufferedReader br = new BufferedReader(new FileReader(inFile));
		String line = null;
		int totalWords = 0;
		while( (line = br.readLine()) != null ) {
			line = line.trim();
			if(! line.isEmpty()) {
				Instance instance = new Instance(this, line, containsLabel);
				instanceList.add(instance);
				totalWords += instance.words.length;
			}
		}
		System.out.println("Total Instances: " + instanceList.size());
		System.out.println("Total Words: " + totalWords);
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
