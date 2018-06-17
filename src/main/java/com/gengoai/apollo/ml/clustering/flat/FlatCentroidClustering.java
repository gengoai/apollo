package com.gengoai.apollo.ml.clustering.flat;

import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.ml.clustering.Clustering;
import com.gengoai.apollo.stat.measure.Measure;

import java.util.List;

/**
 * A clustering that is flat and based on centroids
 *
 * @author David B. Bracewell
 */
public class FlatCentroidClustering extends FlatClustering {
   private static final long serialVersionUID = 1L;

   public FlatCentroidClustering(Clustering other) {
      super(other);
   }

   public FlatCentroidClustering(Clusterer<?> clusterer, Measure measure) {
      super(clusterer, measure);
   }

   protected FlatCentroidClustering(Clusterer<?> clusterer, Measure measure, List<Cluster> clusterList) {
      super(clusterer, measure, clusterList);
   }
}// END OF FlatCentroidClustering
