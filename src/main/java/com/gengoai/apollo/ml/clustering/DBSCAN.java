package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.stat.measure.DistanceMeasure;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class DBSCAN extends FlatClusterer {


   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      DBSCANClusterer<ApacheClusterable> clusterer = new DBSCANClusterer<>(fitParameters.eps,
                                                                           fitParameters.minPts,
                                                                           new ApacheDistanceMeasure(
                                                                              fitParameters.distanceMeasure));
      clusters.clear();
      List<ApacheClusterable> apacheClusterables = dataSupplier.get().map(ApacheClusterable::new).collect();
      List<org.apache.commons.math3.ml.clustering.Cluster<ApacheClusterable>> result = clusterer.cluster(
         apacheClusterables);

      for (int i = 0; i < result.size(); i++) {
         Cluster cp = new Cluster();
         cp.setId(i);
         result.get(i).getPoints().forEach(ap -> cp.addPoint(ap.getVector()));
         clusters.add(cp);
      }
   }

   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters fitParameters) {
      fit(dataSupplier, Cast.as(fitParameters, Parameters.class));
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return new Parameters();
   }

   public static class Parameters extends FitParameters {
      public double eps;
      public int minPts;
      public DistanceMeasure distanceMeasure;
   }

}//END OF DBSCAN
