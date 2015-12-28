package com.davidbracewell.apollo.function;

/**
 * @author David B. Bracewell
 */
public class Sigmoid implements DifferentiableFunction {
  private static final long serialVersionUID = 1L;

  @Override
  public double applyAsDouble(double operand) {
    return 1.0 / (1.0 + Math.exp(-operand));
  }

  @Override
  public double gradientAsDouble(double value) {
    double sigV = applyAsDouble(value);
    return sigV * (1.0 - sigV);
  }


}// END OF Sigmoid
