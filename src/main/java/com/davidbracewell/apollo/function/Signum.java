package com.davidbracewell.apollo.function;

import com.google.common.base.Preconditions;

/**
 * @author David B. Bracewell
 */
public class Signum implements DifferentiableFunction {
  private static final long serialVersionUID = 1L;

  @Override
  public double applyAsDouble(double operand) {
    return Math.signum(operand);
  }

  @Override
  public double gradientAsDouble(double value) {
    Preconditions.checkArgument(value != 0, "Not differentiable at 0");
    return 0;
  }

}// END OF Signum
