package com.davidbracewell.apollo.distribution;

import com.davidbracewell.function.SerializableDoubleUnaryOperator;

/**
 * The interface Density.
 *
 * @author David B. Bracewell
 */
public interface Density extends SerializableDoubleUnaryOperator {

  /**
   * Log apply as double double.
   *
   * @param v the v
   * @return the double
   */
  double logApplyAsDouble(double v);

}// END OF Density
