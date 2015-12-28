package com.davidbracewell.apollo;

import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class TanH implements DifferentiableFunction {

  @Override
  public double applyAsDouble(double operand) {
    return FastMath.tanh(operand);
  }

  @Override
  public double gradientAsDouble(double value) {
    return (1.0 - Math.pow(applyAsDouble(value), 2.0));
  }


}// END OF TanH
