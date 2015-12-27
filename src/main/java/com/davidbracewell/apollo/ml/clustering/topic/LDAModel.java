package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.affinity.Similarity;
import com.davidbracewell.apollo.distribution.ConditionalMultinomial;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class LDAModel extends Clustering {
  private static final long serialVersionUID = 1L;
  ArrayList<Cluster> clusters;
  ConditionalMultinomial wordTopic;
  double alpha;
  double beta;

  /**
   * Instantiates a new Clustering.
   *
   * @param encoderPair the encoder pair
   */
  protected LDAModel(EncoderPair encoderPair) {
    super(encoderPair, Similarity.Cosine.asDistanceMeasure());
  }

  @Override
  public int size() {
    return clusters.size();
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
  public List<Cluster> getClusters() {
    return clusters;
  }

  @Override
  public double[] softCluster(Instance instance) {
    return new double[0];
  }

  public Counter<String> getTopicWords(int topic) {
    Counter<String> counter = Counters.newHashMapCounter();
    for (int i = 0; i < getFeatureEncoder().size(); i++) {
      counter.set(
        getFeatureEncoder().decode(i).toString(),
        wordTopic.probability(topic, i)
      );
    }
    return counter;
  }


}// END OF LDAModel
