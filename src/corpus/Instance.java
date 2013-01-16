package corpus;

import java.util.ArrayList;

public class Instance {
	public int[] words;
	public int label = -1;
	Corpus c;
	public Instance(Corpus c, String line, boolean containsLabel) {//read from line
		String splitted[] = line.split(c.delimiter);
		ArrayList<Integer> tempWordArray = new ArrayList<Integer>(); 
		if(containsLabel) {
			words = new int[splitted.length - 1];
		} else {
			words = new int[splitted.length];
		}
		for(int i=0; i<splitted.length; i++) {
			if(i == splitted.length - 1) { //last field is label
				if(containsLabel) {
					label = c.getLabelMap(splitted[i]);
					continue;
				}
			}
			String word = splitted[i];
			int index = c.corpusVocab.getIndex(word);
			if(index > 0) {
				tempWordArray.add(index);
			}
		}
		if(tempWordArray.size() == 0) {
			//all the features did not pass the threshold, add single UNKNOWN
			tempWordArray.add(0);
		}
		words = new int[tempWordArray.size()];
		for(int i=0; i<words.length; i++) {
			words[i] = tempWordArray.get(i);
		}
	}
}
