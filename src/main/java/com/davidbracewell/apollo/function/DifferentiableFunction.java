package com.davidbracewell.apollo.function;

import com.davidbracewell.function.SerializableDoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public interface DifferentiableFunction extends SerializableDoubleUnaryOperator {
  double H = 0.000001;

  default double gradientAsDouble(double value) {
    return (applyAsDouble(value + H) - applyAsDouble(value)) / H;
  }

}//END OF DifferentiableFunction
