package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.stat.measure.Measure;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public abstract class FlatClusterer implements Clusterer {
   private static final long serialVersionUID = 1L;
   protected final List<Cluster> clusters = new ArrayList<>();
   protected Measure measure;
   protected int numberOfFeatures;

   @Override
   public Measure getMeasure() {
      return measure;
   }

   @Override
   public int getNumberOfFeatures() {
      return numberOfFeatures;
   }

   @Override
   public int getNumberOfLabels() {
      return 0;
   }

   @Override
   public Cluster getCluster(int index) {
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

}//END OF FlatClusterer
