package com.davidbracewell.apollo.ml.clustering.hierarchical;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.DoubleStream;

/**
 * The enum Linkage.
 *
 * @author David B. Bracewell
 */
public enum Linkage {
   /**
    * The Min.
    */
   Min {
      @Override
      public double calculate(@NonNull DoubleStream doubleStream) {
         return doubleStream.min().getAsDouble();
      }
   },
   /**
    * The Max.
    */
   Max {
      @Override
      public double calculate(@NonNull DoubleStream doubleStream) {
         return doubleStream.max().getAsDouble();
      }
   },
   /**
    * The Average.
    */
   Average {
      @Override
      public double calculate(@NonNull DoubleStream doubleStream) {
         return doubleStream.average().getAsDouble();
      }
   };

   /**
    * Calculate double.
    *
    * @param doubleStream the double stream
    * @return the double
    */
   public abstract double calculate(DoubleStream doubleStream);

   /**
    * Calculate double.
    *
    * @param c1              the c 1
    * @param c2              the c 2
    * @param distanceMeasure the distance measure
    * @return the double
    */
   public final double calculate(@NonNull Cluster c1, @NonNull Cluster c2, @NonNull DistanceMeasure distanceMeasure) {
      List<Double> distances = new ArrayList<>();
      for (Vector t1 : flatten(c1)) {
         for (Vector t2 : flatten(c2)) {
            distances.add(distanceMeasure.calculate(t1, t2));
         }
      }
      return calculate(distances.stream().mapToDouble(Double::doubleValue));
   }

   /**
    * Calculate double.
    *
    * @param v               the v
    * @param cluster         the cluster
    * @param distanceMeasure the distance measure
    * @return the double
    */
   public final double calculate(@NonNull Vector v, @NonNull Cluster cluster, @NonNull DistanceMeasure distanceMeasure) {
      return calculate(cluster.getPoints().stream().mapToDouble(v2 -> distanceMeasure.calculate(v, v2)));
   }

   /**
    * Flatten list.
    *
    * @param c the c
    * @return the list
    */
   protected List<Vector> flatten(Cluster c) {
      if (c == null) {
         return Collections.emptyList();
      }
      if (!c.getPoints().isEmpty()) {
         return c.getPoints();
      }
      List<Vector> list = new ArrayList<>();
      list.addAll(flatten(c.getLeft()));
      list.addAll(flatten(c.getRight()));
      return list;
   }


}//END OF Linkage
