package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.clustering.CentroidClustering;
import com.davidbracewell.apollo.ml.clustering.Cluster;

import java.util.List;

/**
 * The type K means clustering.
 *
 * @author David B. Bracewell
 */
public class KMeansClustering extends FlatClustering implements CentroidClustering {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Clustering.
    *
    * @param encoderPair     the encoder pair
    * @param distanceMeasure the distance measure
    */
   public KMeansClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
      super(encoderPair, distanceMeasure);
   }

   /**
    * Instantiates a new K means clustering.
    *
    * @param encoderPair     the encoder pair
    * @param distanceMeasure the distance measure
    * @param clusterList     the cluster list
    */
   public KMeansClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure, List<Cluster> clusterList) {
      super(encoderPair, distanceMeasure, clusterList);
   }
}// END OF KMeansClustering
