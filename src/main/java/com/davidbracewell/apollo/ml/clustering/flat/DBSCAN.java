package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class DBSCAN extends Clusterer<FlatClustering> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private double eps;
   @Getter
   @Setter
   private int minPts;
   @Getter
   @Setter(onParam = @_({@NonNull}))
   private DistanceMeasure distanceMeasure;


   public DBSCAN() {
      this(Distance.Euclidean, 0.01, 2);
   }

   public DBSCAN(@NonNull DistanceMeasure distanceMeasure, double eps, int minPts) {
      this.distanceMeasure = distanceMeasure;
      this.eps = eps;
      this.minPts = minPts;
   }


   @Override
   public FlatClustering cluster(MStream<Vector> instances) {
      DBSCANClusterer<ApacheClusterable> clusterer = new DBSCANClusterer<>(eps, minPts,
                                                                           new ApacheDistanceMeasure(distanceMeasure));

      List<Cluster> clusters = clusterer.cluster(instances.map(ApacheClusterable::new).collect())
                                        .stream()
                                        .map(c -> {
                                           com.davidbracewell.apollo.ml.clustering.Cluster cp = new com.davidbracewell.apollo.ml.clustering.Cluster();
                                           c.getPoints().forEach(ap -> cp.addPoint(ap.getVector()));
                                           return cp;
                                        }).collect(Collectors.toList());
      KMeansClustering clustering = new KMeansClustering(getEncoderPair(), distanceMeasure);
      clusters.forEach(clustering::addCluster);
      return clustering;
   }


}// END OF DBSCAN
