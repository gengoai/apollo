package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.apollo.ml.clustering.flat.FlatClustering;
import com.davidbracewell.apollo.optimization.Optimum;
import com.davidbracewell.collection.counter.Counter;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public abstract class TopicModel extends FlatClustering {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   protected int K;

   public TopicModel(TopicModel other) {
      super(other);
      this.K = other.K;
   }

   public TopicModel(Clusterer<?> clusterer, Measure measure, int k) {
      super(clusterer, measure);
      K = k;
   }

   /**
    * Gets the distribution across topics for a given feature.
    *
    * @param feature the feature (word) whose topic distribution is desired
    * @return the distribution across topics for the given feature
    */
   public abstract double[] getTopicDistribution(String feature);

   /**
    * Gets topic vector.
    *
    * @param topic the topic
    * @return the topic vector
    */
   public abstract Vector getTopicVector(int topic);

   /**
    * Gets the words and their probabilities for a given topic
    *
    * @param topic the topic
    * @return the topic words
    */
   public abstract Counter<String> getTopicWords(int topic);

   @Override
   public int hardCluster(@NonNull Instance instance) {
      return Optimum.MAXIMUM.optimumIndex(softCluster(instance));
   }

   @Override
   public int size() {
      return K;
   }

}// END OF TopicModel
