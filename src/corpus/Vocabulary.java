package corpus;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Vocabulary {
	private static int index = 0;
	public static int vocabSize = -1;
	public static Map<String, Integer> wordToIndex = new HashMap<String, Integer>();
	public static ArrayList<String> indexToWord = new ArrayList<String>();
	public static Map<Integer, Integer> indexToFrequency = new HashMap<Integer, Integer>();
	//public static Map<String, Integer> wordToFrequency = new HashMap<String, Integer>();
	
	public static void readVocabFromFile(String filename, boolean containsLabel) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = null;
		
		while( (line = br.readLine()) != null) {
			line = line.trim();
			if(! line.isEmpty()) {
				String words[] = line.split(Corpus.delimiter);
				for(int i=0; i<words.length; i++) {
					if(i == words.length-1) {
						if(containsLabel) {
							continue;
						}
					}
					String word = words[i];
					if(wordToIndex.containsKey(word)) {
						int wordIndex = wordToIndex.get(word);
						int oldFreq = indexToFrequency.get(wordIndex);
						indexToFrequency.put(wordIndex, oldFreq + 1);
					} else {
						wordToIndex.put(word, index);
						indexToFrequency.put(index, 1);
						index++;
					}
				}
			}
		}
		vocabSize = wordToIndex.size();
		System.out.println("Vocab Size: " + vocabSize);
		br.close();
		
	}
	
	public static int getIndex(String word) {
		if(wordToIndex.containsKey(word)) {
			return wordToIndex.get(word);
		} else {
			return -1;
		}
	}
	
}
