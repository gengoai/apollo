package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Streams;
import com.gengoai.tuple.Tuple2;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public interface Clusterer extends Model, Iterable<Cluster> {
   @Override
   default NDArray estimate(NDArray data) {
      return softCluster(data);
   }

   Cluster getCluster(int id);

   Measure getMeasure();

   Cluster getRoot();

   /**
    * Performs a hard clustering, which determines the single cluster the given instance belongs to
    *
    * @return the index of the cluster that the instance belongs to
    */
   default int hardCluster(NDArray vector) {
      return getMeasure().getOptimum()
                         .optimum(Streams.asParallelStream(this)
                                         .map(c -> $(c.getId(), getMeasure().calculate(vector, c.getCentroid()))),
                                  Tuple2::getV2)
                         .map(Tuple2::getKey)
                         .orElse(-1);
   }

   /**
    * Checks if the clustering is flat
    *
    * @return True if flat, False otherwise
    */
   default boolean isFlat() {
      return false;
   }

   /**
    * Checks if the clustering is hierarchical
    *
    * @return True if hierarchical, False otherwise
    */
   default boolean isHierarchical() {
      return false;
   }

   /**
    * The number of clusters
    *
    * @return the number of clusters
    */
   int size();

   /**
    * Performs a soft clustering, which provides a membership probability of the given instance to the clusters
    *
    * @param instance the instance
    * @return membership probability of the given instance to the clusters
    */
   default NDArray softCluster(NDArray instance) {
      NDArray distances = NDArrayFactory.DENSE.zeros(size()).fill(Double.POSITIVE_INFINITY);
      int assignment = hardCluster(instance);
      if (assignment >= 0) {
         distances.set(assignment, getMeasure().calculate(getCluster(assignment).getCentroid(), instance));
      }
      return distances;
   }


}//END OF Clusterer
