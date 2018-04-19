package com.gengoai.apollo.ml.clustering.hierarchical;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.apollo.ml.clustering.Cluster;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * The enum Linkage.
 *
 * @author David B. Bracewell
 */
public enum Linkage {
   /**
    * Single link, which calculates the minimum distance between elements
    */
   Single {
      @Override
      public double calculate(@NonNull DoubleStream doubleStream) {
         return doubleStream.min().orElse(Double.POSITIVE_INFINITY);
      }
   },
   /**
    * Complete link, which calculates the maximum distance between elements
    */
   Complete {
      @Override
      public double calculate(@NonNull DoubleStream doubleStream) {
         return doubleStream.max().orElse(Double.POSITIVE_INFINITY);
      }
   },
   /**
    * Average link, which calculates the mean distance between elements
    */
   Average {
      @Override
      public double calculate(@NonNull DoubleStream doubleStream) {
         return doubleStream.average().orElse(Double.POSITIVE_INFINITY);
      }
   };

   /**
    * Calculates a value over the given stream of doubles which represents a stream of distances between an instance and
    * a cluster.
    *
    * @param doubleStream the double stream
    * @return the calculated value
    */
   public abstract double calculate(DoubleStream doubleStream);

   /**
    * Calculates the linkage metric between two clusters
    *
    * @param c1              the first cluster
    * @param c2              the second cluster
    * @param distanceMeasure the distance measure to use
    * @return the linkage metric
    */
   public final double calculate(@NonNull Cluster c1, @NonNull Cluster c2, @NonNull Measure distanceMeasure) {
      List<Double> distances = new ArrayList<>();
      for (NDArray t1 : flatten(c1)) {
         distances.addAll(flatten(c2).stream()
                                     .map(t2 -> distanceMeasure.calculate(t1, t2))
                                     .collect(Collectors.toList()));
      }
      return calculate(distances.stream().mapToDouble(Double::doubleValue));
   }

   /**
    * Calculates the linkage metric between a vector and a cluster
    *
    * @param v               the vector
    * @param cluster         the cluster
    * @param distanceMeasure the distance measure to use
    * @return the linkage metric
    */
   public final double calculate(@NonNull NDArray v, @NonNull Cluster cluster, @NonNull Measure distanceMeasure) {
      return calculate(cluster.getPoints().stream().mapToDouble(v2 -> distanceMeasure.calculate(v, v2)));
   }

   /**
    * Flattens a cluster down to a single list of vectors
    *
    * @param c the cluster
    * @return the list of vectors
    */
   protected List<NDArray> flatten(Cluster c) {
      if (c == null) {
         return Collections.emptyList();
      }
      if (!c.getPoints().isEmpty()) {
         return c.getPoints();
      }
      List<NDArray> list = new ArrayList<>();
      list.addAll(flatten(c.getLeft()));
      list.addAll(flatten(c.getRight()));
      return list;
   }


}//END OF Linkage
