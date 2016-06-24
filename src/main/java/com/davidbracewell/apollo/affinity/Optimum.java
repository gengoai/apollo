package com.davidbracewell.apollo.affinity;

import java.util.Comparator;

/**
 * @author David B. Bracewell
 */
public enum Optimum implements Comparator<Double> {
  MINIMUM {
    @Override
    public int compare(double v1, double v2) {
      return Double.compare(v1, v2);
    }

    @Override
    public boolean test(double value, double threshold) {
      return value <= threshold;
    }

    @Override
    public double startingValue() {
      return Double.POSITIVE_INFINITY;
    }

  },
  MAXIMUM {
    @Override
    public int compare(double v1, double v2) {
      return -Double.compare(v1, v2);
    }

    @Override
    public boolean test(double value, double threshold) {
      return value >= threshold;
    }

    @Override
    public double startingValue() {
      return Double.NEGATIVE_INFINITY;
    }
  };

  @Override
  public int compare(Double o1, Double o2) {
    if (o1 == null) return -1;
    if (o2 == null) return 1;
    return compare(o1, o2);
  }


  public abstract int compare(double v1, double v2);

  public abstract boolean test(double value, double threshold);

  public abstract double startingValue();

}//END OF Optimum
