package com.davidbracewell.apollo.ml.clustering;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.util.Arrays;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public interface CentroidClustering extends Clustering {

   @Override
   default int hardCluster(@NonNull Instance instance) {
      Vector vector = instance.toVector(getEncoderPair());
      return getClusters().parallelStream()
                          .map(c -> $(c.getId(), getDistanceMeasure().calculate(vector, c.getCentroid())))
                          .min((t1, t2) -> Double.compare(t1.getValue(), t2.getValue()))
                          .map(Tuple2::getKey)
                          .orElse(-1);
   }

   @Override
   default double[] softCluster(@NonNull Instance instance) {
      double[] distances = new double[size()];
      Arrays.fill(distances, Double.POSITIVE_INFINITY);
      FeatureVector vector = instance.toVector(getEncoderPair());
      int assignment = hardCluster(instance);
      if (assignment >= 0) {
         distances[assignment] = getDistanceMeasure().calculate(get(assignment).getCentroid(), vector);
      }
      return distances;
   }


}// END OF CentroidClustering
