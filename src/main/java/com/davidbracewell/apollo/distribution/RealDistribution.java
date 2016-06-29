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
   * Probability double.
   *
   * @param value the value
   * @return the double
   */
  double probability(double value);

  @Override
  default double applyAsDouble(double operand) {
    return probability(operand);
  }

  @Override
  default double logApplyAsDouble(double v) {
    return Math.log(probability(v));
  }

  /**
   * Cumulative probability double.
   *
   * @param value the value
   * @return the double
   */
  double cumulativeProbability(double value);

  /**
   * Cumulative probability double.
   *
   * @param lowerBound  the lower bound
   * @param higherBound the higher bound
   * @return the double
   */
  double cumulativeProbability(double lowerBound, double higherBound);


  /**
   * Inverse cumulative probability double.
   *
   * @param p the p
   * @return the double
   */
  double inverseCumulativeProbability(double p);


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
