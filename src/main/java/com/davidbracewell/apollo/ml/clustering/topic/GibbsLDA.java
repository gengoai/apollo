package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.distribution.ConditionalMultinomial;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>Basic LDA using Gibbs sampling</p>
 *
 * @author David B. Bracewell
 */
public class GibbsLDA extends Clusterer<LDAModel> {
   private static final long serialVersionUID = 1L;
   private static final Logger log = Logger.getLogger(GibbsLDA.class);

   @Getter
   @Setter
   private int K = 100;
   @Getter
   @Setter
   private double alpha = 0;
   @Getter
   @Setter
   private double beta = 0;
   @Getter
   @Setter
   private int maxIterations = 1000;
   @Getter
   @Setter
   private int burnin = 250;
   @Getter
   @Setter
   private int sampleLag = 25;
   @Getter
   @Setter
   private boolean verbose = true;
   @Getter
   @Setter
   private boolean keepDocumentTopicAssignments = false;
   @Getter
   @Setter(onParam = @_({@NonNull}))
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

   @Override
   public LDAModel cluster(MStream<Vector> instanceStrream) {
      List<Vector> instances = instanceStrream.collect();
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
         Vector vector = instances.get(m);
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
      model.K = K;
      model.randomGenerator = randomGenerator;
      if (sampleLag <= 0) {
         model.wordTopic = nw.copy();
      } else {
         model.wordTopic = new ConditionalMultinomial(K, V, beta);
         for (int w = 0; w < V; w++) {
            for (int k = 0; k < K; k++) {
               model.wordTopic.increment(k, w, (int) (phisum[k].get(w) / numstats));
            }
         }
      }
      model.clusters = new ArrayList<>(K);

      if (keepDocumentTopicAssignments) {
         for (int k = 0; k < K; k++) {
            TopicCluster cluster = new TopicCluster();
            model.clusters.add(cluster);
            for (int m = 0; m < M; m++) {
               double p;
               if (sampleLag <= 0) {
                  p = nd.probability(m, k);
               } else {
                  double c = thetasum[m].get(k) / numstats;
                  p = (c + alpha) / (K * alpha + nd.sum(m));
               }
               if (p > 0) {
                  cluster.addPoint(instances.get(m), p);
               }
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
            thetasum[m].increment(k, nd.count(m, k));
         }
      }
      for (int k = 0; k < K; k++) {
         for (int w = 0; w < V; w++) {
            phisum[k].increment(w, nw.count(k, w));
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


}// END OF GibbsLDA
