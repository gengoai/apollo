package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Cluster;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class FlatSoftClustering extends FlatHardClustering {
  private static final long serialVersionUID = 1L;

  /**
   * Instantiates a new Clustering.
   *
   * @param encoderPair     the encoder pair
   * @param distanceMeasure the distance measure
   */
  public FlatSoftClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
    super(encoderPair, distanceMeasure);
  }

  public FlatSoftClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure, List<Cluster> clusters) {
    super(encoderPair, distanceMeasure, clusters);
  }


  @Override
  public double[] softCluster(Instance instance) {
    Vector distances = new DenseVector(size());
    FeatureVector vector = instance.toVector(getEncoderPair());
    for (int i = 0; i < distances.dimension(); i++) {
      distances.set(i, getDistanceMeasure().calculate(clusters.get(i).getCentroid(), vector));
    }
    double max = distances.max();
    distances.mapSelf(d -> Math.exp(max - d));
    double sum = distances.sum();
    distances.mapSelf(d -> d / sum);
    return distances.toArray();
  }

}// END OF FlatSoftClustering
