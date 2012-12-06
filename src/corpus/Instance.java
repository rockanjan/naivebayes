package corpus;

public class Instance {
	public int[] words;
	public int label = -1;
	Corpus c;
	public Instance(Corpus c, String line, boolean containsLabel) {//read from line
		String splitted[] = line.split(c.delimiter);
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
			words[i] = c.corpusVocab.getIndex(word);
		}
	}
}
