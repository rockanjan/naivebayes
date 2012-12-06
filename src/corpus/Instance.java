package corpus;

public class Instance {
	public int[] words;
	public Instance(String line) {//read from line
		String splitted[] = line.split(Corpus.delimiter);
		words = new int[splitted.length];
		for(int i=0; i<splitted.length; i++) {
			String word = splitted[i];
			words[i] = Vocabulary.getIndex(word);
		}
	}
}
