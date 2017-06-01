package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.affinity.Similarity;
import com.davidbracewell.apollo.optimization.Optimum;
import com.davidbracewell.apollo.distribution.ConditionalMultinomial;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.flat.FlatClustering;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import lombok.NonNull;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;

/**
 * <p>A flat clustering for the result of LDA topic models</p>
 *
 * @author David B. Bracewell
 */
public class LDAModel extends FlatClustering {
   private static final long serialVersionUID = 1L;
   ArrayList<Cluster> clusters;
   ConditionalMultinomial wordTopic;
   int K;
   RandomGenerator randomGenerator;
   double alpha;
   double beta;

   /**
    * Instantiates a new LDAModel.
    *
    * @param encoderPair the encoder pair
    */
   LDAModel(EncoderPair encoderPair) {
      super(encoderPair, Similarity.Cosine.asDistanceMeasure());
   }

   @Override
   public int size() {
      return K;
   }

   @Override
   public Cluster get(int index) {
      return clusters.get(index);
   }

   @Override
   public Cluster getRoot() {
      return clusters.get(0);
   }

   @Override
   public int hardCluster(@NonNull Instance instance) {
      return Optimum.MAXIMUM.optimum(softCluster(instance)).getV1();
   }

   @Override
   public double[] softCluster(@NonNull Instance instance) {
      Vector vector = instance.toVector(getEncoderPair());
      int[] docTopic = new int[K];
      SparseVector docWordTopic = new SparseVector(getEncoderPair().numberOfFeatures());

      for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
         int topic = randomGenerator.nextInt(K);
         docTopic[topic]++;
         docWordTopic.set(entry.index, topic);
         wordTopic.increment(topic, entry.index);
      }

      for (int iteration = 0; iteration < 100; iteration++) {
         for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
            int topic = sample(entry.getIndex(), (int) docWordTopic.get(entry.index), docTopic);
            docWordTopic.set(entry.index, topic);
         }
      }

      double[] p = new double[K];
      for (int i = 0; i < K; i++) {
         p[i] = (docTopic[i] + alpha) / (docTopic.length + K * alpha);
      }

      for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
         wordTopic.decrement((int) docWordTopic.get(entry.index), entry.index);
      }

      return p;
   }

   private int sample(int word, int topic, int[] nd) {
      wordTopic.decrement(topic, word);
      nd[topic]--;

      double[] p = new double[K];
      for (int k = 0; k < K; k++) {
         p[k] = wordTopic.probability(k, word) * (nd[k] + alpha) / (nd.length - 1 + K * alpha);
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


      wordTopic.increment(topic, word);
      nd[topic]++;

      return topic;
   }

   /**
    * Gets the words and their probabilities for a given topic
    *
    * @param topic the topic
    * @return the topic words
    */
   public Counter<String> getTopicWords(int topic) {
      Counter<String> counter = Counters.newCounter();
      for (int i = 0; i < getFeatureEncoder().size(); i++) {
         counter.set(getFeatureEncoder().decode(i).toString(),
                     wordTopic.probability(topic, i));
      }
      return counter;
   }


   /**
    * Gets the distribution across topics for a given feature.
    *
    * @param feature the feature (word) whose topic distribution is desired
    * @return the distribution across topics for the given feature
    */
   public double[] getTopicDistribution(String feature) {
      double[] distribution = new double[clusters.size()];
      int i = (int) getFeatureEncoder().encode(feature);
      if (i == -1) {
         return distribution;
      }
      for (int k = 0; k < clusters.size(); k++) {
         distribution[k] = wordTopic.probability(k, i);
      }
      return distribution;
   }


}// END OF LDAModel
