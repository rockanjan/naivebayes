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
	private int index = 0;
	public int vocabSize = -1;
	public Map<String, Integer> wordToIndex = new HashMap<String, Integer>();
	public ArrayList<String> indexToWord = new ArrayList<String>();
	public Map<Integer, Integer> indexToFrequency = new HashMap<Integer, Integer>();
	//public static Map<String, Integer> wordToFrequency = new HashMap<String, Integer>();
	
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
	
	public void readVocabFromFile(Corpus c, String filename, boolean containsLabel) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = null;
		
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
		System.out.println("Vocab Size: " + vocabSize);
		br.close();
		
	}
	
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
			return -1;
		}
	}
	
}
