package com.davidbracewell.apollo.ml.clustering;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public interface Clusterer<T extends Clusterable, C extends Clustering<T>> {

  /**
   * Clusters a number of instances
   *
   * @param instances The instances to cluster
   * @return The result of the clustering
   */
  C cluster(List<? extends T> instances);

}// END OF Clusterer
