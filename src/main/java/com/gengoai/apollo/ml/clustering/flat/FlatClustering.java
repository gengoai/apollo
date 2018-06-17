package com.gengoai.apollo.ml.clustering.flat;

import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.ml.clustering.Clustering;
import com.gengoai.apollo.stat.measure.Measure;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Base class for flat clusterings
 *
 * @author David B. Bracewell
 */
public abstract class FlatClustering extends Clustering {
   private static final long serialVersionUID = 1L;
   /**
    * The Clusters.
    */
   protected final List<Cluster> clusters = new ArrayList<>();

   public FlatClustering(Clustering other) {
      super(other);
   }

   /**
    * Instantiates a new Flat clustering.
    *
    * @param clusterer the clusterer
    * @param measure   the measure
    */
   public FlatClustering(Clusterer<?> clusterer, Measure measure) {
      super(clusterer, measure);
   }

   /**
    * Instantiates a new Flat clustering.
    *
    * @param clusterer   the clusterer
    * @param measure     the measure
    * @param clusterList the cluster list
    */
   protected FlatClustering(Clusterer<?> clusterer, Measure measure, List<Cluster> clusterList) {
      this(clusterer, measure);
      this.clusters.addAll(clusterList);
      for (int i = 0; i < clusters.size(); i++) {
         clusters.get(i).setId(i);
      }
   }

   /**
    * Adds a cluster to the clustering.
    *
    * @param cluster the cluster
    */
   public void addCluster(@NonNull Cluster cluster) {
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
   public Iterator<Cluster> iterator() {
      return clusters.iterator();
   }

   @Override
   public int size() {
      return clusters.size();
   }

}// END OF FlatClustering
