package corpus;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Corpus {
	public static String delimiter = "\\+";
	public static InstanceList instanceList = new InstanceList();
	
	public static void read(String inFile) throws IOException {
		Vocabulary.readVocabFromFile(inFile);
		BufferedReader br = new BufferedReader(new FileReader(inFile));
		String line = null;
		int totalWords = 0;
		while( (line = br.readLine()) != null ) {
			line = line.trim();
			if(! line.isEmpty()) {
				Instance instance = new Instance(line);
				instanceList.add(instance);
				totalWords += instance.words.length;
			}
		}
		System.out.println("Total Instances: " + instanceList.size());
		System.out.println("Total Words: " + totalWords);
		br.close();
	}
	public static void main(String[] args) throws IOException {
		String inFile = "/home/anjan/workspace/SRL-anjan/myconll2005/final/nbayes/combined.final.propprocessed.span";
		Corpus.read(inFile);
	}
}
