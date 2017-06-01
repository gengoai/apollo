package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.optimization.Optimum;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.flat.FlatClustering;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;

/**
 * The type Lsa model.
 *
 * @author David B. Bracewell
 */
public class LSAModel extends FlatClustering {

   /**
    * The number of topics.
    */
   int K;

   /**
    * Instantiates a new Lsa model.
    *
    * @param encoderPair     the encoder pair
    * @param distanceMeasure the distance measure
    */
   protected LSAModel(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
      super(encoderPair, distanceMeasure);
   }

   /**
    * Gets the number of topics.
    *
    * @return the number of topics.
    */
   public int getK() {
      return K;
   }

   /**
    * Gets topic vector.
    *
    * @param topic the topic
    * @return the topic vector
    */
   public Vector getTopicVector(int topic) {
      return get(topic)
                .getPoints()
                .get(0);
   }

   /**
    * Gets the words and their probabilities for a given topic
    *
    * @param topic the topic
    * @return the topic words
    */
   public Counter<String> getTopicWords(int topic) {
      Vector v = getTopicVector(topic);
      Counter<String> c = Counters.newCounter();
      v.forEachSparse(e -> c.set(decodeFeature(e.getIndex()).toString(), e.getValue()));
      return c;
   }

   @Override
   public int hardCluster(Instance instance) {
      return Optimum.MAXIMUM.optimumIndex(softCluster(instance));
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
