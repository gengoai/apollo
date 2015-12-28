package com.davidbracewell.apollo.function;

/**
 * @author David B. Bracewell
 */
public class Linear implements DifferentiableFunction {
  private static final long serialVersionUID = 1L;

  @Override
  public double applyAsDouble(double operand) {
    return operand;
  }

  @Override
  public double gradientAsDouble(double value) {
    return 1;
  }
}// END OF Linear
