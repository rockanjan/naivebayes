package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;

import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;
import corpus.Vocabulary;

public class NBayes {
	Corpus c;
	double smoothParam = 1; // smoothing parameter for normalization
	int K = -1;
	int V = -1;
	int N = -1;

	double[] pi;
	double[][] emission; // emission[w][Y] == P(w | Y), size V * K
	double[][] posterior; // posterior[Y][Xi] = P(Y | Xi), size K * N

	double[] piExpectedCounts;
	double[][] emissionExpectedCounts;

	boolean containsLabel;

	double MAX_EMISSION_EXP = -Double.MAX_VALUE;

	double oldLogLikelihood = 0;
	double logLikelihood = 0;

	public static String base = "out/model/";
	private String outFolderPrefix;
	public static String modelFolder;
	
	public NBayes(Corpus c, int K, int V, boolean containsLabel, String outFolderPrefix) {
		this.containsLabel = containsLabel;
		this.c = c;
		this.K = K;
		this.V = V;
		this.N = c.trainInstanceList.size();
		this.outFolderPrefix = outFolderPrefix;
		modelFolder = base + outFolderPrefix + "/";

	}

	public void initializeSupervised() {
		pi = new double[K];
		emission = new double[V][K];
		for (int n = 0; n < N; n++) {
			if(n % 100000 == 0)	System.out.println("Sentence Number: " + n);
			for (int k = 0; k < K; k++) {
				if (c.trainInstanceList.get(n).label == -1) {
					System.err
							.println("Error: labeled instance does not have label assigned");
					System.exit(1);
				}
				if (c.trainInstanceList.get(n).label == k) {
					pi[k]++;
				}
			}
		}
		double sum = 0;
		for (int k = 0; k < K; k++) {
			sum += pi[k];
		}
		// normalize
		for (int k = 0; k < K; k++) {
			pi[k] = Math.log(pi[k] / sum);
		}
		
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
//				for (int v = 0; v < V; v++) {
					Instance instance = c.trainInstanceList.get(n);
					for (int i = 0; i < instance.words.length; i++) {
//						if (instance.words[i] == v && instance.label == k) {
						if (instance.label == k) {
							emission[instance.words[i]][k]++;
//						}
					}
				}
			}
		}

		for (int k = 0; k < K; k++) {
			sum = 0;
			for (int v = 0; v < V; v++) {
				emission[v][k] += smoothParam;
				sum += emission[v][k];
			}
			// normalize
			for (int v = 0; v < V; v++) {
				emission[v][k] = Math.log(emission[v][k] / sum);
				if (emission[v][k] > MAX_EMISSION_EXP) {
					MAX_EMISSION_EXP = emission[v][k];
				}
			}
		}

		sanityCheck();
	}

	public void initializeRandom(Random r) {
		pi = new double[K];
		emission = new double[V][K];

		double sum = 0;
		for (int k = 0; k < pi.length; k++) {
			pi[k] = r.nextDouble() + 0.01;
			sum += pi[k];
		}
		// normalize
		for (int k = 0; k < pi.length; k++) {
			pi[k] = Math.log(pi[k] / sum);
			// System.out.println("pi " + Math.exp(pi[k]));
		}

		for (int k = 0; k < K; k++) {
			sum = 0;
			for (int v = 0; v < V; v++) {
				emission[v][k] = r.nextDouble() + 0.01;
				sum += emission[v][k];
			}
			// normalize
			for (int v = 0; v < V; v++) {
				emission[v][k] = Math.log(emission[v][k] / sum);
				// System.out.println(Math.exp(emission[v][k]));
				if (emission[v][k] > MAX_EMISSION_EXP) {
					MAX_EMISSION_EXP = emission[v][k];
				}
			}
		}

		sanityCheck();
	}

	public void train(int numIter) {
		int iterCount = 0;
		System.out.println("Starting EM");
		int smallCount = 0;
		DecimalFormat df = new DecimalFormat("#.###");
		while (iterCount < numIter) {
			oldLogLikelihood = logLikelihood;
			logLikelihood = 0;
			long startTime = System.currentTimeMillis();
			
			//e-step
			long eStartTime = System.currentTimeMillis();
			eStep();
			long eEndTime = System.currentTimeMillis();
			String eTime = (1.0 * (eEndTime - eStartTime) / 1000 / 60) + " minutes";
			//System.out.println("\t E-step time : " + eTime);
			
			//m-step
			long mStartTime = System.currentTimeMillis();
			mStep();
			long mEndTime = System.currentTimeMillis();
			String mTime = (1.0 * (mEndTime - mStartTime) / 1000 / 60) + " minutes";
			//System.out.println("\t M-step time : " + mTime);
			
			
			long endTime = System.currentTimeMillis();
			String time = df.format(1.0 * (endTime - startTime) / 1000 / 60)
					+ " minutes";
			double diff = logLikelihood - oldLogLikelihood;
			System.out.println("Itr " + ++iterCount + " LL = " + logLikelihood
					+ " \tdiff = " + diff
					+ "\t time " + time);
			sanityCheck();
			if(Math.abs(diff / oldLogLikelihood) < 1E-6) {
				smallCount ++;
			} else {
				smallCount = 0;
			}
			if(smallCount == 3) {
				System.out.println("diff ratio : " + (-diff/oldLogLikelihood));
				System.out.println("Exiting because LL has converged");
				break;
			}
		}
	}

	//e-step overall complexity O(K N |X_i|) where |X_i| is the average token length of the instance
	public void eStep() {
		// find the posteriors P(Y=k | X=n; \theta)
		initPosterior();
		for (int n = 0; n < N; n++) {
			double denominator = 0.0;
			double sumexp = 0.0;
			for (int k = 0; k < K; k++) {
				posterior[k][n] = computeJoint(n, k); // log (p(y,x))
			}				
			for (int k= 0; k < K; k++) {
				sumexp += Math.exp(posterior[k][n] - MAX_EMISSION_EXP);
			}
			denominator = MAX_EMISSION_EXP + Math.log(sumexp); // log(p(x))
			logLikelihood += denominator;
			for (int k= 0; k < K; k++) {
				double ratio = posterior[k][n] - denominator;
				// System.out.println(ratio);
				double prob = Math.exp(ratio);
				if (prob == 0) {
					System.err.println("Numerator = " + posterior[k][n] + " Denominator = "
							+ denominator + " Ratio = " + ratio + " Prob = " + prob);
					System.exit(-1);
				}
				// returns actual probability
				posterior[k][n] = prob; 
			}
		}

		// compute expected counts
		piExpectedCounts = new double[K];
		emissionExpectedCounts = new double[V][K];
		for (int k = 0; k < K; k++) {
			for (int n = 0; n < N; n++) {
				piExpectedCounts[k] += posterior[k][n];
			}
		}

		for (int k = 0; k < K; k++) {
			//for (int v = 0; v < V; v++) {
				for (int n = 0; n < N; n++) {
					Instance instance = c.trainInstanceList.get(n);
					for (int i = 0; i < instance.words.length; i++) {
						//if (instance.words[i] == v) {
							emissionExpectedCounts[instance.words[i]][k] += posterior[k][n];
						//}
					}
				}
			//}
		}
	}

	public void mStep() {
		// normalize
		// smoothing
		double sum = 0;
		for (int k = 0; k < K; k++) {
			piExpectedCounts[k] += smoothParam;
			sum += piExpectedCounts[k];
		}
		verifyPiSum(sum);// sum should be equal to N + K * smoothParam

		// normalize prior
		for (int k = 0; k < K; k++) {
			pi[k] = Math.log(piExpectedCounts[k] / sum);
		}

		// normalize emission
		MAX_EMISSION_EXP = -Double.MAX_VALUE;
		for (int k = 0; k < K; k++) {
			sum = 0;
			for (int v = 0; v < V; v++) {
				emissionExpectedCounts[v][k] += smoothParam;
				sum += emissionExpectedCounts[v][k];
			}
			for (int v = 0; v < V; v++) {
				emission[v][k] = Math.log(emissionExpectedCounts[v][k] / sum);
				// System.out.println(emission[v][k]);
				if (emission[v][k] > MAX_EMISSION_EXP) {
					MAX_EMISSION_EXP = emission[v][k];
				}
			}
		}
		clearExpectedCounts();
	}

	public void sanityCheck() {
		double piSum = 0;
		for (int k = 0; k < K; k++) {
			piSum += Math.exp(pi[k]);
		}
		if (Math.abs(piSum - 1) > 1E-3) {
			System.err.println("Error: Sanity check of pi not OK, sum = "
					+ piSum);
			System.exit(-1);
		}

		for (int k = 0; k < K; k++) {
			double emissionSum = 0;
			for (int v = 0; v < V; v++) {
				emissionSum += Math.exp(emission[v][k]);
			}
			if (Math.abs(piSum - 1) > 1E-3) {
				System.err
						.println("Error: Sanity check of emission for cluster = "
								+ k + " not OK, sum = " + emissionSum);
				System.exit(-1);
			}
		}
	}

	public void clearExpectedCounts() {
		piExpectedCounts = null;
		for (int v = 0; v < V; v++) {
			emissionExpectedCounts[v] = null;
		}
		emissionExpectedCounts = null;
	}

	private void verifyPiSum(double sum) {
		double actual = (N + K * smoothParam);
		if (Math.abs(sum - actual) > 1E-3) {
			System.err.println("Error: pi sum = " + sum + " should be "
					+ actual);
		}
	}

	/**
	 * @return joint probability in log
	 * @param n
	 *            = instance number
	 * @param k
	 *            = class
	 */
	public double computeJoint(int n, int k) {
		Instance instance = c.trainInstanceList.get(n);
		double prior = pi[k];
		double prob = 0.0;
		for (int i = 0; i < instance.words.length; i++) {
			prob += emission[instance.words[i]][k];
		}
		prob = prior + prob;
		return prob;
	}

	public double computeJointTest(int n, int k) {
		Instance instance = c.testInstanceList.get(n);
		double prior = pi[k];
		double prob = 0.0;
		for (int i = 0; i < instance.words.length; i++) {
			prob += emission[instance.words[i]][k];
		}
		prob = prior + prob;
		return prob;
	}

	public void initPosterior() {
		posterior = new double[K][N];
	}

	public void clearPosterior() {
		for (int i = 0; i < K; i++) {
			posterior[i] = null;
		}
		posterior = null;
	}

	public void save() throws FileNotFoundException {
		File file = new File(modelFolder);
		if (!file.exists()) {
			boolean success = file.mkdirs();
			if (success) {
				System.out.println("Model output folder created.");
			} else {
				System.out.println("Error creating model output folder.");
			}
		}
		while(!file.isDirectory()) {
			System.err.println("The model output folder is not a directory");
			System.out.print("Enter new model base path: ");
			BufferedReader br = new BufferedReader(new InputStreamReader(
					System.in));
			try {
				modelFolder = br.readLine();
			} catch (IOException ioe) {
				System.out.println("IO error trying to read the base path!");
				System.exit(1);
			}
			
			file = new File(modelFolder);
			if(!file.exists()) {
				boolean success = file.mkdirs();
				if (!success) {
					System.out.println("Model output folder created.");
				}
			}
		}
		// dictionary
		PrintWriter dictionaryWriter = new PrintWriter(modelFolder + "/dictionary.txt");
		dictionaryWriter.println(V);
		for (int v = 0; v < V; v++) {
			dictionaryWriter.println(c.corpusVocab.indexToWord.get(v));
			dictionaryWriter.flush();
		}
		dictionaryWriter.close();

		// prior
		PrintWriter piWriter = new PrintWriter(modelFolder + "/pi.txt");
		piWriter.println(K);
		for (int k = 0; k < K; k++) {
			piWriter.println(pi[k]);
			piWriter.flush();
		}
		piWriter.close();

		// emission
		PrintWriter emissionWriter = new PrintWriter(modelFolder + "/emission.txt");
		emissionWriter.println(V);
		for (int v = 0; v < V; v++) {
			for (int k = 0; k < K; k++) {
				emissionWriter.println(emission[v][k]);
				emissionWriter.flush();
			}
		}
		emissionWriter.close();
		
		if(containsLabel) {
			//labels
			PrintWriter labelWriter = new PrintWriter(modelFolder + "/label.txt");
			labelWriter.println(c.labelMap.size());
			for(int i=0; i<c.labelIdToString.size(); i++) {
				labelWriter.println(c.labelIdToString.get(i));
				labelWriter.flush();
			}
			labelWriter.close();
		}
		
	}

	public void load() throws IOException {
		/*
		//dictionary
		System.out.println("\treading dictionary...");
		c.readVocabFromVocabFile(base + "/dictionary.txt");
		*/		
		System.out.println("\treading prior...");
		BufferedReader brPi = new BufferedReader(new FileReader(modelFolder + "/pi.txt"));
		String line = null;
		line = brPi.readLine().trim();
		K = Integer.parseInt(line);
		pi = new double[K];
		int index = 0;
		while( (line = brPi.readLine()) != null ) {
			pi[index++] = Double.parseDouble(line);			
		}
		if(index != K) {
			System.err.println("Prior distribution file corrupted.");
			System.exit(-1);			
		}
		brPi.close();
		
		BufferedReader brEmission = new BufferedReader(new FileReader(modelFolder + "/emission.txt"));
		line = brEmission.readLine().trim();
		V = Integer.parseInt(line);
		emission = new double[V][K];
		index = 0;
		MAX_EMISSION_EXP = -Double.MAX_VALUE;
		while( (line = brEmission.readLine()) != null) {
			int v = (int) index/K;
			int k = index % K;
			emission[v][k] = Double.parseDouble(line);
			if(emission[v][k] > MAX_EMISSION_EXP) {
				MAX_EMISSION_EXP = emission[v][k];
			}
			index++;
		}
		brEmission.close();
		System.out.println("\treading emission...");
		//sanity check
		sanityCheck();
		
		/*
		if(containsLabel) {
			System.out.println("\treading labels...");
			BufferedReader brLabel = new BufferedReader(new FileReader(modelFolder + "/label.txt"));
			line = brLabel.readLine().trim();
			c.labelIdToString = new ArrayList<String>();
			while( (line = brLabel.readLine() ) != null) {
				line = line.trim();
				c.labelIdToString.add(line);
			}
			brLabel.close();
		}
		*/
	}

	public void decode(String outFile) throws FileNotFoundException {
		System.out.println("Decoding to " + outFile);
		PrintWriter pw = new PrintWriter(outFile);
		for (int n = 0; n < c.testInstanceList.size(); n++) {
			int maxCluster = -1;
			double maxProb = -Double.MAX_VALUE;
			for (int k = 0; k < K; k++) {
				double prob = computeJointTest(n, k);
				// System.out.println("prob " + Math.exp(prob));
				if (prob > maxProb) {
					maxProb = prob;
					maxCluster = k;
				}
			}
			pw.println(maxCluster);
			pw.flush();
		}
		pw.close();
		System.out.println("Decoding completed.");
	}

	public void decodeLabeled(String outFile) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(outFile);
		int correct = 0;
		System.out
				.println("Instance \t Actual \t Predicted \t error \tprobability");
		int totalB = 0;
		int correctB = 0;
		for (int n = 0; n < c.testInstanceList.size(); n++) {
			int maxCluster = -1;
			double maxProb = -Double.MAX_VALUE;
			double maxActualProb = -Double.MAX_VALUE;
			for (int k = 0; k < K; k++) {
				double prob = Math.exp(computeJointTest(n, k));
				double actualProb = prob;
				double denom = 0;
				for (int i = 0; i < K; i++) {
					denom += Math.exp(computeJointTest(n, i));
				}
				actualProb = prob / denom;
				//System.out.println("prob " + prob + " actual prob " + actualProb);
				if (prob > maxProb) {
					maxProb = prob;
					maxCluster = k;
					maxActualProb = actualProb;
				}
			}
			//System.out.println();
			String label = c.labelIdToString.get(maxCluster);
			pw.println(label);
			int actualCluster = c.testInstanceList.get(n).label;
			
			if(c.labelIdToString.get(actualCluster).equals("B")) {
				totalB++;
				if(c.labelIdToString.get(maxCluster).equals("B")) {
					correctB++;
				}
			}
			
			if (maxCluster == actualCluster) {
				/*System.out.println((n + 1) + "\t\t"
						+ c.labelIdToString.get(actualCluster) + "\t\t" + label
						+ "\t\t" + " " + "\t\t"
						+ new DecimalFormat("##.##").format(maxActualProb));
						*/
				correct++;				
			} else {
				/*System.out.println((n + 1) + "\t\t"
						+ c.labelIdToString.get(actualCluster) + "\t\t" + label
						+ "\t\t" + "*" + "\t\t"
						+ new DecimalFormat("##.##").format(maxActualProb));
						*/
			}
						
			pw.flush();
		}
		System.out.println("CorrectB = " + correctB + " totalB = " + totalB + " Accuray for B = " + 100.0 * correctB / totalB);
		System.out.println("Correct = " + correct + " Total = " + c.testInstanceList.size()
				+ " Accuracy = " + (100.0 * correct / c.testInstanceList.size()));
		pw.close();
	}
	
	public void decodeLabeledVector(String outFile) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(outFile);
		int correct = 0;
		int totalB = 0;
		int correctB = 0;
		DecimalFormat df = new DecimalFormat("#.#");
		int[] keys = {c.labelMap.get("B"), c.labelMap.get("I"), c.labelMap.get("O")};
		for (int n = 0; n < c.testInstanceList.size(); n++) {
			int maxCluster = -1;
			double maxProb = -Double.MAX_VALUE;
			double maxActualProb = -Double.MAX_VALUE;
			int printCount = 0;
			for (int k : keys) {
				double prob = Math.exp(computeJointTest(n, k));
				double actualProb = prob;
				double denom = 0;
				for (int i = 0; i < K; i++) {
					denom += Math.exp(computeJointTest(n, i));
				}
				actualProb = prob / denom;
				pw.print(df.format(actualProb));
				printCount++;
				if(printCount == keys.length) {
					pw.println();
				} else {
					pw.print(" ");
				}
				//System.out.println("prob " + prob + " actual prob " + actualProb);
				if (prob > maxProb) {
					maxProb = prob;
					maxCluster = k;
					maxActualProb = actualProb;
				}
			}
			//System.out.println();
			String label = c.labelIdToString.get(maxCluster);
			int actualCluster = c.testInstanceList.get(n).label;
			if(c.labelIdToString.get(actualCluster).equals("B")) {
				totalB++;
				if(c.labelIdToString.get(maxCluster).equals("B")) {
					correctB++;
				}
			}
			if (maxCluster == actualCluster)
				correct++;
			pw.flush();
		}
		System.out.println("CorrectB = " + correctB + " totalB = " + totalB + " Accuray for B = " + 100.0 * correctB / totalB);
		System.out.println("Correct = " + correct + " Total = " + c.testInstanceList.size()
				+ " Accuracy = " + (100.0 * correct / c.testInstanceList.size()));
		pw.close();
	}
}
