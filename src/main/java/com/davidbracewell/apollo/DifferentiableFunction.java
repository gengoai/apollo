package com.davidbracewell.apollo;

import com.davidbracewell.function.SerializableDoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public interface DifferentiableFunction extends SerializableDoubleUnaryOperator {
  double H = 0.0001;
  double TWO_H = 2.0 * H;

  default double gradientAsDouble(double value) {
    return (applyAsDouble(value + H) - applyAsDouble(value - H)) / TWO_H;
  }

}//END OF DifferentiableFunction
