package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.affinity.Similarity;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clustering;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class LDAModel extends Clustering {
  private static final long serialVersionUID = 1L;

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
    return 0;
  }

  @Override
  public Cluster get(int index) {
    return null;
  }

  @Override
  public Cluster getRoot() {
    return null;
  }

  @Override
  public List<Cluster> getClusters() {
    return null;
  }

  @Override
  public double[] softCluster(Instance instance) {
    return new double[0];
  }

}// END OF LDAModel
