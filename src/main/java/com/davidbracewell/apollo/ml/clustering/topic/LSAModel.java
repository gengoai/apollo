package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;

/**
 * The type Lsa model.
 *
 * @author David B. Bracewell
 */
public class LSAModel extends TopicModel {

   /**
    * Instantiates a new Lsa model.
    *
    * @param encoderPair     the encoder pair
    * @param distanceMeasure the distance measure
    */
   protected LSAModel(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
      super(encoderPair, distanceMeasure);
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
   public Vector getTopicVector(int topic) {
      return get(topic)
                .getPoints()
                .get(0);
   }

   @Override
   public Counter<String> getTopicWords(int topic) {
      Vector v = getTopicVector(topic);
      Counter<String> c = Counters.newCounter();
      v.forEachSparse(e -> c.set(decodeFeature(e.getIndex()).toString(), e.getValue()));
      return c;
   }

   @Override
   public double[] softCluster(Instance instance) {
      double[] scores = new double[size()];
      Vector v = instance.toVector(getEncoderPair());
      for (int i = 0; i < size(); i++) {
         double score = v.dot(getTopicVector(i));
         scores[i] = score;
      }
      return scores;
   }

}// END OF LSAModel
