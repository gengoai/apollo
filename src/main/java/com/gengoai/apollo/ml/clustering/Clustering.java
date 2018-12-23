package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Streams;
import com.gengoai.tuple.Tuple2;

import java.io.Serializable;
import java.util.Arrays;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public abstract class Clustering implements Serializable, Iterable<Cluster> {
   private final Measure measure;

   protected Clustering(Measure measure) {
      this.measure = measure;
   }


   /**
    * Gets the  cluster for the given index.
    *
    * @param index the index
    * @return the cluster
    */
   public abstract Cluster get(int index);

   /**
    * Gets the root of the hierarchical cluster.
    *
    * @return the root
    */
   public Cluster getRoot() {
      throw new UnsupportedOperationException();
   }

   /**
    * Performs a hard clustering, which determines the single cluster the given instance belongs to
    *
    * @param instance the instance
    * @return the index of the cluster that the instance belongs to
    */
   public int hardCluster(NDArray instance) {
      return measure.getOptimum()
                    .optimum(Streams.asParallelStream(this)
                                    .map(c -> $(c.getId(), measure.calculate(instance, c.getCentroid()))),
                             Tuple2::getV2)
                    .map(Tuple2::getKey)
                    .orElse(-1);
   }

   /**
    * Checks if the clustering is flat
    *
    * @return True if flat, False otherwise
    */
   public boolean isFlat() {
      return false;
   }

   /**
    * Checks if the clustering is hierarchical
    *
    * @return True if hierarchical, False otherwise
    */
   public boolean isHierarchical() {
      return false;
   }

   /**
    * The number of clusters
    *
    * @return the number of clusters
    */
   public abstract int size();

   /**
    * Performs a soft clustering, which provides a membership probability of the given instance to the clusters
    *
    * @param instance the instance
    * @return membership probability of the given instance to the clusters
    */
   public double[] softCluster(NDArray instance) {
      double[] distances = new double[size()];
      Arrays.fill(distances, Double.POSITIVE_INFINITY);
      int assignment = hardCluster(instance);
      if (assignment >= 0) {
         distances[assignment] = measure.calculate(get(assignment).getCentroid(), instance);
      }
      return distances;
   }
}//END OF Clustering
