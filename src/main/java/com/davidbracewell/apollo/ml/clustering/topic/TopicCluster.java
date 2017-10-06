package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.clustering.Cluster;

import java.util.HashMap;
import java.util.Map;

/**
 * Specialized cluster to allow fast lookup of vector scores.
 *
 * @author David B. Bracewell
 */
class TopicCluster extends Cluster {
   private static final long serialVersionUID = 1L;
   private final Map<NDArray, Double> scores = new HashMap<>();

   /**
    * Adds a point (vector) to the cluster
    *
    * @param vector the vector to add
    * @param score  the score of the vector
    */
   public void addPoint(NDArray vector, double score) {
      super.addPoint(vector);
      scores.put(vector, score);
   }

   @Override
   public void clear() {
      super.clear();
      scores.clear();
   }

   @Override
   public double getScore(NDArray point) {
      return scores.getOrDefault(point, 0.0);
   }

}// END OF TopicCluster
