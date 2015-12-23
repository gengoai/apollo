package com.davidbracewell.apollo.ml.clustering.hierarchical;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.DoubleStream;

/**
 * @author David B. Bracewell
 */
public enum Linkage {
  Min {
    @Override
    public double calculate(@NonNull DoubleStream doubleStream) {
      return doubleStream.min().getAsDouble();
    }
  },
  Max {
    @Override
    public double calculate(@NonNull DoubleStream doubleStream) {
      return doubleStream.max().getAsDouble();
    }
  },
  Average {
    @Override
    public double calculate(@NonNull DoubleStream doubleStream) {
      return doubleStream.average().getAsDouble();
    }
  };

  public abstract double calculate(DoubleStream doubleStream);

  public final double calculate(@NonNull Cluster c1, @NonNull Cluster c2, @NonNull DistanceMeasure distanceMeasure) {
    List<Double> distances = new ArrayList<>();
    for (FeatureVector t1 : flatten(c1)) {
      for (FeatureVector t2 : flatten(c2)) {
        distances.add(distanceMeasure.calculate(t1, t2));
      }
    }
    return calculate(distances.stream().mapToDouble(Double::doubleValue));
  }

  public final double calculate(@NonNull Vector v, @NonNull Cluster cluster, @NonNull DistanceMeasure distanceMeasure) {
    return calculate(cluster.getPoints().stream().mapToDouble(v2 -> distanceMeasure.calculate(v, v2)));
  }

  protected List<FeatureVector> flatten(Cluster c) {
    if (c == null) {
      return Collections.emptyList();
    }
    if (!c.getPoints().isEmpty()) {
      return c.getPoints();
    }
    List<FeatureVector> list = new ArrayList<>();
    list.addAll(flatten(c.getLeft()));
    list.addAll(flatten(c.getRight()));
    return list;
  }


}//END OF Linkage
