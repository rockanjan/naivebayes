package corpus;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Vocabulary {
	private int featureThreshold = 50;
	//index zero reserved for __OOV__ (low freq features)
	private int index = 1;
	public int vocabSize = -1;
	public String UNKNOWN = "_UNKNOWN_";
	public Map<String, Integer> wordToIndex = new HashMap<String, Integer>();
	public ArrayList<String> indexToWord = new ArrayList<String>();
	public Map<Integer, Integer> indexToFrequency = new HashMap<Integer, Integer>();
	
	private void addItem(String word) {
		if(wordToIndex.containsKey(word)) {
			int wordIndex = wordToIndex.get(word);
			int oldFreq = indexToFrequency.get(wordIndex);
			indexToFrequency.put(wordIndex, oldFreq + 1);
		} else {
			wordToIndex.put(word, index);
			indexToWord.add(word);
			indexToFrequency.put(index, 1);
			index++;
		}
	}
	
	private void reduceVocab() {
		Map<String, Integer> wordToIndexNew = new HashMap<String, Integer>();
		ArrayList<String> indexToWordNew = new ArrayList<String>();
		Map<Integer, Integer> indexToFrequencyNew = new HashMap<Integer, Integer>();
		wordToIndexNew.put(UNKNOWN, 0);
		indexToFrequencyNew.put(0, -1); //TODO: decide if this matters
		indexToWordNew.add(UNKNOWN);
		int featureIndex = 1;
		for(int i=1; i<indexToWord.size(); i++) {
			if(indexToFrequency.get(i) > featureThreshold) {
				wordToIndexNew.put(indexToWord.get(i), featureIndex);
				indexToWordNew.add(indexToWord.get(i));
				indexToFrequencyNew.put(featureIndex, indexToFrequency.get(i));
				featureIndex = featureIndex + 1;
			}
		}
		indexToWord = null; indexToFrequency = null; wordToIndex = null;
		indexToWord = indexToWordNew;
		indexToFrequency = indexToFrequencyNew;
		wordToIndex = wordToIndexNew;
		vocabSize = wordToIndex.size();
		System.out.println("New vocab size : " + vocabSize);
		
	}
	
	public void readVocabFromFile(Corpus c, String filename, boolean containsLabel) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = null;
		wordToIndex.put(UNKNOWN, 0);
		indexToFrequency.put(0, -1); //TODO: decide if this matters
		indexToWord.add(UNKNOWN);
		while( (line = br.readLine()) != null) {
			line = line.trim();
			if(! line.isEmpty()) {
				String words[] = line.split(c.delimiter);
				for(int i=0; i<words.length; i++) {
					if(i == words.length-1) {
						if(containsLabel) {
							continue;
						}
					}
					String word = words[i];
					addItem(word);
				}
			}
		}
		vocabSize = wordToIndex.size();
		System.out.println("Original Vocab Size: " + vocabSize);
		if(featureThreshold > 0) {
			System.out.println("Reducing vocab size with threshold : " + featureThreshold);
			reduceVocab();
		}
		br.close();
		
	}
	
	//reads from the dictionary
	public void readVocabFromVocabFile(String filename) {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		String line = null;
		try {
			line = br.readLine().trim();
			vocabSize = Integer.parseInt(line);
			while( (line = br.readLine()) != null) {
				line = line.trim();
				if(line.isEmpty()) {
					continue;
				}
				addItem(line);
			}
		} catch (IOException e) {
			e.printStackTrace();
			System.err.println("error reading vocab file");
		}
		if(vocabSize != wordToIndex.size()) {
			System.out.println("Vocab file corrputed: header size and the vocab size do not match");
			System.exit(-1);
		}
	}
	
	public int getIndex(String word) {
		if(wordToIndex.containsKey(word)) {
			return wordToIndex.get(word);
		} else {
			//word not found in vocab
			System.out.println(word + " not found in vocab");
			return 0; //unknown id
		}
	}
	
}
