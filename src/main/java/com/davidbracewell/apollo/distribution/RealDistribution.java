package com.davidbracewell.apollo.distribution;

import com.google.common.base.Preconditions;

/**
 * The interface Real distribution.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface RealDistribution<T extends RealDistribution> extends Density {

  /**
   * Sample double.
   *
   * @return the double
   */
  double sample();

  /**
   * Sample double [ ].
   *
   * @param size the size
   * @return the double [ ]
   */
  default double[] sample(int size) {
    Preconditions.checkArgument(size > 0, "Size must be > 0");
    double[] samples = new double[size];
    for (int i = 0; i < size; i++) {
      samples[i] = sample();
    }
    return samples;
  }


  /**
   * Increment t.
   *
   * @param value the value
   * @return the t
   */
  T increment(double value);


  /**
   * Decrement t.
   *
   * @param value the value
   * @return the t
   */
  default T decrement(double value) {
    return increment(-value);
  }


}//END OF RealDistribution
