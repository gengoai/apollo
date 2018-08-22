package com.gengoai.apollo.ml.clustering.topic;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.stat.distribution.ConditionalMultinomial;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Iterables;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import lombok.NonNull;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.List;

/**
 * <p>A flat clustering for the result of LDA topic models</p>
 *
 * @author David B. Bracewell
 */
public class LDAModel extends TopicModel {
   private static final long serialVersionUID = 1L;
   ConditionalMultinomial wordTopic;
   RandomGenerator randomGenerator;
   double alpha;
   double beta;

   public LDAModel(TopicModel other) {
      super(other);
   }

   public LDAModel(Clusterer<?> clusterer, Measure measure, int k) {
      super(clusterer, measure, k);
   }

   @Override
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

   @Override
   public NDArray getTopicVector(int topic) {
      return NDArrayFactory.wrap(wordTopic.probabilities(topic));
   }

   @Override
   public Counter<String> getTopicWords(int topic) {
      Counter<String> counter = Counters.newCounter();
      for (int i = 0; i < getFeatureEncoder().size(); i++) {
         counter.set(getFeatureEncoder().decode(i).toString(),
                     wordTopic.probability(topic, i));
      }
      return counter;
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

   protected void setClusters(List<Cluster> clusters) {
      this.clusters.clear();
      this.clusters.addAll(clusters);
   }

   @Override
   public double[] softCluster(@NonNull Instance instance) {
      NDArray vector = getPreprocessors().apply(instance).toVector(getEncoderPair());
      int[] docTopic = new int[K];
      NDArray docWordTopic = NDArrayFactory.DEFAULT().zeros(getEncoderPair().numberOfFeatures());

      for (NDArray.Entry entry : Iterables.asIterable(vector.sparseIterator())) {
         int topic = randomGenerator.nextInt(K);
         docTopic[topic]++;
         docWordTopic.set(entry.matrixIndex(), topic);
         wordTopic.increment(topic, entry.matrixIndex());
      }

      for (int iteration = 0; iteration < 100; iteration++) {
         for (NDArray.Entry entry : Iterables.asIterable(vector.sparseIterator())) {
            int topic = sample(entry.matrixIndex(), (int) docWordTopic.get(entry.matrixIndex()), docTopic);
            docWordTopic.set(entry.matrixIndex(), topic);
         }
      }

      double[] p = new double[K];
      for (int i = 0; i < K; i++) {
         p[i] = (docTopic[i] + alpha) / (docTopic.length + K * alpha);
      }

      for (NDArray.Entry entry : Iterables.asIterable(vector.sparseIterator())) {
         wordTopic.decrement((int) docWordTopic.get(entry.matrixIndex()), entry.matrixIndex());
      }

      return p;
   }


}// END OF LDAModel
