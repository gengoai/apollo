package com.gengoai.apollo.ml.clustering;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public interface Clustering extends Serializable, Iterable<Cluster> {

   /**
    * Gets the  cluster for the given index.
    *
    * @param index the index
    * @return the cluster
    */
   Cluster get(int index);

   /**
    * Gets the root of the hierarchical cluster.
    *
    * @return the root
    */
   default Cluster getRoot() {
      throw new UnsupportedOperationException();
   }


   /**
    * Checks if the clustering is flat
    *
    * @return True if flat, False otherwise
    */
   boolean isFlat();

   /**
    * Checks if the clustering is hierarchical
    *
    * @return True if hierarchical, False otherwise
    */
   boolean isHierarchical();

   /**
    * The number of clusters
    *
    * @return the number of clusters
    */
   int size();


}//END OF Clustering
