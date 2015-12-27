package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.distribution.ConditionalMultinomial;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.io.Resources;
import com.davidbracewell.logging.Logger;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Gibbs lda.
 *
 * @author David B. Bracewell
 */
public class GibbsLDA extends Clusterer<LDAModel> {
  private static final Logger log = Logger.getLogger(GibbsLDA.class);

  private int K = 20;
  private double alpha = 0;
  private double beta = 0;
  private int maxIterations = 1000;
  private int burnin = 250;
  private int sampleLag = 25;
  private boolean verbose = true;
  private RandomGenerator randomGenerator = new Well19937c();


  private ConditionalMultinomial nw;
  private ConditionalMultinomial nd;
  private int V;
  private int M;
  private int[][] documents;
  private int[][] z;
  private Vector[] thetasum;
  private Vector[] phisum;
  private int numstats = 0;

  /**
   * The entry point of application.
   *
   * @param args the input arguments
   */
  public static void main(String[] args) {
    Dataset<Instance> d = null;
    try {
      d = Dataset.classification()
        .source(
          Resources.fromFile("/home/david/analysis/working/corpus.txt").lines()
            .map(l -> Instance.create(
              Stream.of(l.toLowerCase().split("\\p{Z}+")).filter(s -> !s.contains("pronoun") && (s.contains("noun") || s.contains("verb") || s.contains("adjective") || s.contains("adverb"))).distinct().map(Feature::TRUE).collect(Collectors.toList())
              )
            )
        ).build();
    } catch (IOException e) {
      e.printStackTrace();
    }
    LDAModel model = new GibbsLDA().train(d);
    for (int k = 0; k < model.wordTopic.getN(); k++) {
      Counter<String> words = model.getTopicWords(k);
      System.out.println(words.topN(10).itemsByCount(false));
    }
  }

  @Override
  public LDAModel cluster(List<LabeledVector> instances) {
    V = getEncoderPair().numberOfFeatures();
    M = instances.size();

    double oAlpha = alpha;
    double oBeta = beta;

    if (alpha <= 0) {
      alpha = 50d / K;
    }


    if (beta <= 0) {
      beta = 200d / V;
    }

    nw = new ConditionalMultinomial(K, V, beta);
    nd = new ConditionalMultinomial(M, K, alpha);
    z = new int[M][];
    documents = new int[M][];

    if (sampleLag > 0) {
      thetasum = new Vector[M];
      for (int m = 0; m < M; m++) {
        thetasum[m] = new SparseVector(K);
      }
      phisum = new Vector[K];
      for (int k = 0; k < K; k++) {
        phisum[k] = new SparseVector(V);
      }
    }


    for (int m = 0; m < M; m++) {
      LabeledVector vector = instances.get(m);
      int N = vector.size();
      z[m] = new int[N];
      documents[m] = new int[N];
      int index = 0;
      for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
        documents[m][index] = entry.getIndex();
        int topic = randomGenerator.nextInt(K);
        z[m][index] = topic;
        nw.increment(topic, entry.index);
        nd.increment(m, topic);
        index++;
      }
    }

    long changed = 0;
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      changed = 0;
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < documents[m].length; n++) {
          int topic = sample(m, n);
          if (z[m][n] != topic) {
            changed++;
          }
          z[m][n] = topic;
        }
      }

      if (iteration > burnin && sampleLag > 0 && ((iteration - burnin) % sampleLag == 0)) {
        updateParams();
      }

      if (verbose && (iteration < 10 || iteration % 50 == 0)) {
        log.info("Iteration {0}: {1} total words changed topics.", iteration, changed);
      }

      if (changed == 0) {
        break;
      }
    }

    if (verbose) {
      log.info("Iteration {0}: {1} total words changed topics.", maxIterations, changed);
    }

    LDAModel model = new LDAModel(getEncoderPair());
    model.alpha = alpha;
    model.beta = beta;
    model.docTopic = nd.copy();
    model.wordTopic = nw.copy();
    model.clusters = new ArrayList<>(K);
    for (int k = 0; k < K; k++) {
      TopicCluster cluster = new TopicCluster();
      model.clusters.add(cluster);
      for (int m = 0; m < M; m++) {
        double p = nd.probability(m, k);
        if (p > 0) {
          cluster.addPoint(instances.get(m), p);
        }
      }
    }


    alpha = oAlpha;
    beta = oBeta;

    return model;
  }

  private void updateParams() {
    for (int m = 0; m < M; m++) {
      for (int k = 0; k < K; k++) {
        thetasum[m].increment(k, nd.probability(m, k));
      }
    }
    for (int k = 0; k < K; k++) {
      for (int w = 0; w < V; w++) {
        phisum[k].increment(w, nw.probability(k, w));
      }
    }
    numstats++;
  }

  private int sample(int m, int n) {
    int topic = z[m][n];
    int wid = documents[m][n];
    nw.decrement(topic, wid);
    nd.decrement(m, topic);

    double[] p = new double[K];
    for (int k = 0; k < K; k++) {
      p[k] = nw.probability(k, wid) * nd.probability(m, k);
      if (k > 0) {
        p[k] += p[k - 1];
      }

    }

    double u = randomGenerator.nextDouble() * p[K - 1];
    for (topic = 0; topic < p.length - 1; topic++) {
      if (u < p[topic]) {
        break;
      }
    }


    nw.increment(topic, wid);
    nd.increment(m, topic);

    return topic;
  }

  /**
   * Gets k.
   *
   * @return the k
   */
  public int getK() {
    return K;
  }

  /**
   * Sets k.
   *
   * @param k the k
   */
  public void setK(int k) {
    K = k;
  }

  /**
   * Gets alpha.
   *
   * @return the alpha
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alpha.
   *
   * @param alpha the alpha
   */
  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  /**
   * Gets beta.
   *
   * @return the beta
   */
  public double getBeta() {
    return beta;
  }

  /**
   * Sets beta.
   *
   * @param beta the beta
   */
  public void setBeta(double beta) {
    this.beta = beta;
  }

  @Override
  public void reset() {
    super.reset();
    nw = null;
    nd = null;
    V = 0;
    M = 0;
    documents = null;
    z = null;
    thetasum = null;
    phisum = null;
    numstats = 0;
  }

  /**
   * Gets random generator.
   *
   * @return the random generator
   */
  public RandomGenerator getRandomGenerator() {
    return randomGenerator;
  }

  /**
   * Sets random generator.
   *
   * @param randomGenerator the random generator
   */
  public void setRandomGenerator(RandomGenerator randomGenerator) {
    this.randomGenerator = randomGenerator;
  }

  /**
   * Is verbose boolean.
   *
   * @return the boolean
   */
  public boolean isVerbose() {
    return verbose;
  }

  /**
   * Sets verbose.
   *
   * @param verbose the verbose
   */
  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

  /**
   * Gets sample lag.
   *
   * @return the sample lag
   */
  public int getSampleLag() {
    return sampleLag;
  }

  /**
   * Sets sample lag.
   *
   * @param sampleLag the sample lag
   */
  public void setSampleLag(int sampleLag) {
    this.sampleLag = sampleLag;
  }

  /**
   * Gets burnin.
   *
   * @return the burnin
   */
  public int getBurnin() {
    return burnin;
  }

  /**
   * Sets burnin.
   *
   * @param burnin the burnin
   */
  public void setBurnin(int burnin) {
    this.burnin = burnin;
  }

  /**
   * Gets max iterations.
   *
   * @return the max iterations
   */
  public int getMaxIterations() {
    return maxIterations;
  }

  /**
   * Sets max iterations.
   *
   * @param maxIterations the max iterations
   */
  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }
}// END OF GibbsLDA
