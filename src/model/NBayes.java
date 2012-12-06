package model;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Date;
import java.util.Random;

import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;
import corpus.Vocabulary;

public class NBayes {
	Corpus c;
	double smoothParam = 1; //smoothing parameter for normalization
	int K = -1;
	int V = -1;
	int N = -1;
			
	double[] pi;
	double[][] emission; //emission[w][Y] == P(w | Y), size V * K
	double[][] posterior; //posterior[Y][Xi] = P(Y | Xi), size K * N
	
	double[] piExpectedCounts;
	double[][] emissionExpectedCounts;
	
	boolean containsLabel;
	
	double MAX_EMISSION_EXP = -Double.MAX_VALUE;
	
	double oldLogLikelihood = 0;
	double logLikelihood = 0;
	
	public NBayes(Corpus c, int K, int V, boolean containsLabel) {
		this.containsLabel = containsLabel;
		this.c = c;
		this.K = K;
		this.V = V;
		this.N = c.trainInstanceList.size();
	}
	
	public void initializeSupervised() {
		pi = new double[K];
		emission = new double[V][K];
		for(int n=0; n<N; n++) {
			for(int k=0; k<K; k++) {
				if(c.trainInstanceList.get(n).label == -1) {
					System.err.println("Error: labeled instance does not have label assigned");
					System.exit(1);
				}
				if(c.trainInstanceList.get(n).label == k) {
					pi[k]++;
				}
			}
		}
		double sum = 0;
		for(int k=0; k<K; k++) {
			sum += pi[k];
		}
		//normalize
		for(int k=0; k<K; k++) {
			pi[k] = Math.log(pi[k] / sum);			
		}
		
		for(int n=0; n<N; n++) {
			for(int k=0; k<K; k++) {
				for(int v=0; v<V; v++) {
					Instance instance = c.trainInstanceList.get(n);
					for(int i=0; i < instance.words.length; i++) {
						if(instance.words[i] == v && instance.label == k) {
							emission[v][k]++;
						}
					}
				}
			}
		}
		
		for(int k=0; k<K; k++) {
			sum = 0;
			for(int v=0; v<V; v++) {
				emission[v][k] += smoothParam;
				sum += emission[v][k];
			}
			//normalize
			for(int v=0; v<V; v++) {
				emission[v][k] = Math.log(emission[v][k] / sum);
				if(emission[v][k] > MAX_EMISSION_EXP) {
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
		for(int k=0; k<pi.length; k++) {
			pi[k] = r.nextDouble() + 0.01;
			sum += pi[k];
		}
		//normalize
		for(int k=0; k<pi.length; k++) {
			pi[k] = Math.log(pi[k] / sum);
			//System.out.println("pi " + Math.exp(pi[k]));
		}
		
		for(int k=0; k<K; k++) {
			sum = 0;
			for(int v=0; v<V; v++) {
				emission[v][k] = r.nextDouble() + 0.01;
				sum += emission[v][k];
			}
			//normalize
			for(int v=0; v<V; v++) {
				emission[v][k] = Math.log(emission[v][k] / sum);
				//System.out.println(Math.exp(emission[v][k]));
				if(emission[v][k] > MAX_EMISSION_EXP) {
					MAX_EMISSION_EXP = emission[v][k];
				}
			}
		}
		
		sanityCheck();
	}
	
	public void train(int numIter){
		int iterCount = 0;
		System.out.println("Starting EM");
		while(iterCount < numIter) {
			oldLogLikelihood = logLikelihood;
			logLikelihood = 0;
			long startTime = System.currentTimeMillis();
			eStep();
			mStep();
			long endTime = System.currentTimeMillis();
			String time = (1.0 * (endTime - startTime)/1000/60) + " minutes";
			System.out.println("Itr " + ++iterCount + " LL = " + logLikelihood + " \tdiff = " + (logLikelihood - oldLogLikelihood) + "\t time " + time + " mins");
			sanityCheck();
		}
	}
	
	public void eStep() {
		//find the posteriors P(Y=k | X=n; \theta)
		initPosterior();
		for(int n=0; n<N; n++) {
			for(int k=0; k<K; k++) {
				posterior[k][n] = computePosterior(n, k);
				//System.out.println("Posterior: " + posterior[k][n]);
			}
		}
		
		//compute expected counts
		piExpectedCounts = new double[K];
		emissionExpectedCounts = new double[V][K];
		for(int k=0; k<K; k++) {
			for(int n=0; n<N; n++) {
				piExpectedCounts[k] += posterior[k][n];
			}
		}
		
		for(int k=0; k<K; k++) {
			for(int v=0; v<V; v++) {
				for(int n=0; n<N; n++) {
					Instance instance = c.trainInstanceList.get(n);
					for(int i=0; i<instance.words.length; i++) {
						if(instance.words[i] == v) {
							emissionExpectedCounts[v][k] += posterior[k][n]; 
						}
					}
				}
			}
		}
	}
	
	public void mStep() {
		//normalize
		//smoothing
		double sum = 0;
		for(int k=0; k<K; k++) {
			piExpectedCounts[k] += smoothParam;
			sum += piExpectedCounts[k]; 
		}
		verifyPiSum(sum);//sum should be equal to N + K * smoothParam
		
		//normalize prior
		for(int k=0; k<K; k++) {
			pi[k] = Math.log(piExpectedCounts[k] / sum);
		}
		
		//normalize emission
		MAX_EMISSION_EXP = -Double.MAX_VALUE;
		for(int k=0; k<K; k++) {
			sum = 0;
			for(int v=0; v<V; v++) {
				emissionExpectedCounts[v][k] += smoothParam;
				sum += emissionExpectedCounts[v][k]; 
			}
			for(int v=0; v<V; v++) {
				emission[v][k] = Math.log(emissionExpectedCounts[v][k] / sum);
				//System.out.println(emission[v][k]);
				if(emission[v][k] > MAX_EMISSION_EXP) {
					MAX_EMISSION_EXP = emission[v][k];
				}
			}
		}
		clearExpectedCounts();
	}
	
	public void sanityCheck() {
		double piSum = 0;
		for(int k=0; k<K; k++) {
			piSum += Math.exp(pi[k]);
		}
		if(Math.abs(piSum - 1) > 1E-5) {
			System.err.println("Error: Sanity check of pi not OK, sum = " + piSum);
		}
		
		
		for(int k=0; k<K; k++) {
			double emissionSum = 0;
			for(int v=0; v<V; v++) {
				emissionSum += Math.exp(emission[v][k]);
			}
			if(Math.abs(piSum - 1) > 1E-5) {
				System.err.println("Error: Sanity check of emission for cluster = " + k + " not OK, sum = " + emissionSum);
			}
		}
	}
	
	public void clearExpectedCounts() {
		piExpectedCounts = null;
		for(int v=0; v<V; v++) {
			emissionExpectedCounts[v] = null;
		}
		emissionExpectedCounts = null;
	}
	
	private void verifyPiSum(double sum) {
		double actual = (N + K * smoothParam);
		if(Math.abs(sum - actual) > 1E-5) {
			System.err.println("Error: pi sum = " + sum + " should be " + actual);
		}
	}
	
	/**
	 * @return Computes posterior probability P(Y = k | X_n), no log
	 * @param n = instance number
	 * @param k = class
	 */
	public double computePosterior(int n, int k) {
		double numerator = computeJoint(n, k); //log (p(y,x))
		//System.out.println("numerator : " + Math.exp(numerator));
		
		double denominator = 0.0;
		double sumexp = 0.0;
		for(int j=0; j<K; j++) {
			sumexp += Math.exp(computeJoint(n, j) - MAX_EMISSION_EXP);
		}
		denominator = MAX_EMISSION_EXP + Math.log(sumexp); //log(p(x))
		logLikelihood += denominator;
		double ratio = numerator - denominator;
		//System.out.println(ratio);
		double prob = Math.exp(ratio);
		if(prob == 0) {
			System.err.println("Numerator = " + numerator + " Denominator = " + denominator + " Ratio = " + ratio + " Prob = " + prob);
			System.exit(-1);
		}
		//returns actual probability
		return prob;
	}	
	
	/**
	 * @return joint probability in log
	 * @param n = instance number
	 * @param k = class
	 */
	public double computeJoint(int n, int k) {
		Instance instance = c.trainInstanceList.get(n);
		double prior = pi[k];
		double prob = 0.0;
		for(int i=0; i<instance.words.length; i++) {
			prob += emission[instance.words[i]][k];
		}
		prob = prior + prob;
		return prob;
	}
	
	public double computeJointDecode(int n, int k) {
		Instance instance = c.decodeInstanceList.get(n);
		double prior = pi[k];
		double prob = 0.0;
		for(int i=0; i<instance.words.length; i++) {
			prob += emission[instance.words[i]][k];
		}
		prob = prior + prob;
		return prob;
	}
	
	public void initPosterior() {
		posterior = new double[K][N];
	}
	
	public void clearPosterior() {
		for(int i=0; i<K; i++) {
			posterior[i] = null;
		}
		posterior = null;
	}
	
	public void save() throws FileNotFoundException {
		String base = "out/model/";
		//dictionary
		PrintWriter dictionaryWriter = new PrintWriter(base + "dictionary.txt");
		dictionaryWriter.println(V);
		for(int v=0; v<V; v++) {
			dictionaryWriter.println(c.corpusVocab.indexToWord.get(v));
			dictionaryWriter.flush();
		}
		dictionaryWriter.close();
		
		//prior
		PrintWriter piWriter = new PrintWriter(base + "pi.txt");
		piWriter.println(K);
		for(int k=0; k<K; k++) {
			piWriter.println(pi[k]);
			piWriter.flush();
		}
		piWriter.close();
		
		//emission
		PrintWriter emissionWriter = new PrintWriter(base + "emission.txt");
		emissionWriter.println(V);
		for(int v=0; v<V; v++) {
			for(int k=0; k<K; k++) {
				emissionWriter.println(emission[v][k]);
				emissionWriter.flush();
			}
		}
		emissionWriter.close();
	}
	
	public void decode(String outFile) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(outFile);
		for(int n=0; n<c.decodeInstanceList.size(); n++) {
			int maxCluster = -1;
			double maxProb = -Double.MAX_VALUE;
			for(int k=0; k<K; k++) {
				double prob = computeJointDecode(n, k);
				//System.out.println("prob " + Math.exp(prob));
				if(prob > maxProb) {
					maxProb = prob;
					maxCluster = k;
				}
			}
			pw.println(maxCluster);
			pw.flush();
		}
		pw.close();
	}
	
	public void decodeLabeled(String outFile) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(outFile);
		int correct = 0;
		System.out.println("Instance \t Actual \t Predicted \t error \tprobability");
		for(int n=0; n<c.decodeInstanceList.size(); n++) {
			int maxCluster = -1;
			double maxProb = -Double.MAX_VALUE;
			double maxActualProb = -Double.MAX_VALUE;
			for(int k=0; k<K; k++) {
				double prob = Math.exp(computeJointDecode(n, k));
				double actualProb = prob;
				double denom = 0;
				for(int i=0; i<K; i++) {
					denom += Math.exp(computeJointDecode(n, i));
				}
				actualProb = prob / denom;
				//System.out.println("prob " + prob + " actual prob " + actualProb);
				if(prob > maxProb) {
					maxProb = prob;
					maxCluster = k;
					maxActualProb = actualProb;
				}
			}
			String label = c.labelIdToString.get(maxCluster);
			pw.println(label);
			int actualCluster = c.decodeInstanceList.get(n).label;
			if(maxCluster == actualCluster) {
				System.out.println((n+1) +"\t\t" + c.labelIdToString.get(actualCluster) + "\t\t" + label + "\t\t" + " " + "\t\t" + 
						new DecimalFormat("##.##").format(maxActualProb));
				correct++;
			} else {
				System.out.println((n+1) +"\t\t" + c.labelIdToString.get(actualCluster) + "\t\t" + label + "\t\t" + "*" + "\t\t" + 
						new DecimalFormat("##.##").format(maxActualProb));
			}
			pw.flush();
		}
		System.out.println("Correct = " + correct + " Total = " + N + " Accuracy = " + (100.0 * correct/N));		
		pw.close();
	}
}
