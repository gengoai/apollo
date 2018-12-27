package com.gengoai.apollo.ml.clustering;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Base class for flat clusterings
 *
 * @author David B. Bracewell
 */
public class FlatClustering implements Clustering {
   private static final long serialVersionUID = 1L;
   private final List<Cluster> clusters = new ArrayList<>();

   /**
    * Adds a cluster to the clustering.
    *
    * @param cluster the cluster
    */
   public void add(Cluster cluster) {
      cluster.setId(this.clusters.size());
      this.clusters.add(cluster);
   }

   @Override
   public Cluster get(int index) {
      return clusters.get(index);
   }


   @Override
   public Cluster getRoot() {
      throw new UnsupportedOperationException();
   }


   @Override
   public boolean isFlat() {
      return true;
   }

   @Override
   public boolean isHierarchical() {
      return false;
   }

   @Override
   public Iterator<Cluster> iterator() {
      return clusters.iterator();
   }

   @Override
   public int size() {
      return clusters.size();
   }

}// END OF FlatClustering
