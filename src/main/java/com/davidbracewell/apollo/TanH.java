package com.davidbracewell.apollo;

/**
 * @author David B. Bracewell
 */
public class TanH implements DifferentiableFunction {
  @Override
  public double applyAsDouble(double operand) {
    return 2.0 / (1.0 + Math.exp(-operand)) - 1.0;
  }

  @Override
  public double gradientAsDouble(double value) {
    return 1.0 - (value * value);
  }


}// END OF TanH
