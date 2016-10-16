package com.davidbracewell.apollo.affinity;

import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.util.Comparator;

/**
 * The enum Optimum.
 *
 * @author David B. Bracewell
 */
public enum Optimum implements Comparator<Double> {
  /**
   * The Minimum.
   */
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
  /**
   * The Maximum.
   */
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

  /**
   * Compare int.
   *
   * @param v1 the v 1
   * @param v2 the v 2
   * @return the int
   */
  public abstract int compare(double v1, double v2);

  /**
   * Test boolean.
   *
   * @param value     the value
   * @param threshold the threshold
   * @return the boolean
   */
  public abstract boolean test(double value, double threshold);

  /**
   * Starting value double.
   *
   * @return the double
   */
  public abstract double startingValue();

  /**
   * Select best tuple 2.
   *
   * @param array the array
   * @return tuple 2
   */
  public Tuple2<Integer,Double> selectBest(@NonNull double[] array){
    int bestIndex = selectBestIndex(array);
    return Tuple2.of(bestIndex, array[bestIndex]);
  }

  /**
   * Select best value double.
   *
   * @param array the array
   * @return the double
   */
  public double selectBestValue(@NonNull double[] array) {
    double val = startingValue();
    for (double anArray : array) {
      if (test(anArray, val)) {
        val = anArray;
      }
    }
    return val;
  }

  /**
   * Select best index int.
   *
   * @param array the array
   * @return the int
   */
  public int selectBestIndex(@NonNull double[] array) {
    double val = startingValue();
    int index = -1;
    for (int i = 0; i < array.length; i++) {
      if (test(array[i], val)) {
        val = array[i];
        index = i;
      }
    }
    return index;
  }


}//END OF Optimum
