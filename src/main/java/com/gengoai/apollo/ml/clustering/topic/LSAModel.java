package com.gengoai.apollo.ml.clustering.topic;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;

/**
 * The type Lsa model.
 *
 * @author David B. Bracewell
 */
public class LSAModel extends TopicModel {

   public LSAModel(TopicModel other) {
      super(other);
   }

   public LSAModel(Clusterer<?> clusterer, Measure measure, int k) {
      super(clusterer, measure, k);
   }

   @Override
   public double[] getTopicDistribution(String feature) {
      int i = getFeatureEncoder().index(feature);
      if (i == -1) {
         return new double[K];
      }
      double[] dist = new double[K];
      for (int i1 = 0; i1 < K; i1++) {
         dist[i1] = get(i1).getPoints().get(0).get(i);
      }
      return dist;
   }

   @Override
   public NDArray getTopicVector(int topic) {
      return get(topic)
                .getPoints()
                .get(0);
   }

   @Override
   public Counter<String> getTopicWords(int topic) {
      NDArray v = getTopicVector(topic);
      Counter<String> c = Counters.newCounter();
      v.forEachSparse(e -> c.set(decodeFeature(e.getIndex()).toString(), e.getValue()));
      return c;
   }

   @Override
   public double[] softCluster(Instance instance) {
      double[] scores = new double[size()];
      NDArray vector = getPreprocessors().apply(instance).toVector(getEncoderPair());
      for (int i = 0; i < size(); i++) {
         double score = vector.dot(getTopicVector(i));
         scores[i] = score;
      }
      return scores;
   }

}// END OF LSAModel
