package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.clustering.CentroidClustering;
import com.davidbracewell.apollo.ml.clustering.Cluster;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class KMeansClustering extends FlatClustering implements CentroidClustering {
  /**
   * Instantiates a new Clustering.
   *
   * @param encoderPair     the encoder pair
   * @param distanceMeasure the distance measure
   */
  public KMeansClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
    super(encoderPair, distanceMeasure);
  }

  public KMeansClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure, List<Cluster> clusterList) {
    super(encoderPair, distanceMeasure, clusterList);
  }
}// END OF KMeansClustering
