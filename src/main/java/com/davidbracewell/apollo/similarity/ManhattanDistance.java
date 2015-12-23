package com.davidbracewell.apollo.similarity;

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;

import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class ManhattanDistance extends DistanceMeasure {

  @Override
  public double calculate(Map<?, ? extends Number> m1, Map<?, ? extends Number> m2) {
    Preconditions.checkNotNull(m1, "Vectors cannot be null");
    Preconditions.checkNotNull(m2, "Vectors cannot be null");
    double sum = 0;
    for (Object o : Sets.union(m1.keySet(), m2.keySet())) {
      double d1 = m1.containsKey(o) ? m1.get(o).doubleValue() : 0d;
      double d2 = m2.containsKey(o) ? m2.get(o).doubleValue() : 0d;
      sum += Math.abs(d1 - d2);
    }
    return sum;
  }

  @Override
  public double calculate(double[] v1, double[] v2) {
    Preconditions.checkNotNull(v1, "Vectors cannot be null");
    Preconditions.checkNotNull(v2, "Vectors cannot be null");
    Preconditions.checkArgument(v1.length == v2.length, "Dimension mismatch " + v1.length + " != " + v2.length);
    double sum = 0;
    for (int i = 0; i < v1.length; i++) {
      sum += Math.abs(v1[i] - v2[i]);
    }
    return sum;
  }

}//END OF ManhattanDistance
