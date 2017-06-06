package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Base class for flat clusterings
 *
 * @author David B. Bracewell
 */
public abstract class FlatClustering implements Clustering, Serializable {
   private static final long serialVersionUID = 1L;
   protected final List<Cluster> clusters = new ArrayList<>();
   private final DistanceMeasure distanceMeasure;
   private EncoderPair encoderPair;

   /**
    * Instantiates a new flat clustering.
    *
    * @param encoderPair     the encoder pair
    * @param distanceMeasure the distance measure
    */
   protected FlatClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
      this.encoderPair = encoderPair;
      this.distanceMeasure = distanceMeasure;
   }

   /**
    * Instantiates a new Flat clustering.
    *
    * @param encoderPair     the encoder pair
    * @param distanceMeasure the distance measure
    * @param clusterList     the cluster list
    */
   protected FlatClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure, List<Cluster> clusterList) {
      this(encoderPair, distanceMeasure);
      this.clusters.addAll(clusterList);
      for (int i = 0; i < clusters.size(); i++) {
         clusters.get(i).setId(i);
      }
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
   public DistanceMeasure getDistanceMeasure() {
      return distanceMeasure;
   }

   @Override
   public int size() {
      return clusters.size();
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
   public EncoderPair getEncoderPair() {
      return encoderPair;
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
   public Iterator<Cluster> iterator() {
      return clusters.iterator();
   }
}// END OF FlatClustering
