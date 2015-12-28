package com.davidbracewell.apollo;

/**
 * @author David B. Bracewell
 */
public class Sigmoid implements DifferentiableFunction {
  @Override
  public double applyAsDouble(double operand) {
    return 1.0 / (1.0 + Math.exp(-operand));
  }

  @Override
  public double gradientAsDouble(double value) {
    double sig = applyAsDouble(value);
    return sig * (1.0 - sig);
  }
}// END OF Sigmoid
