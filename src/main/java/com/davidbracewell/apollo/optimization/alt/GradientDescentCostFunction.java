package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value
public class GradientDescentCostFunction implements CostFunction {
   LossFunction lossFunction;
   Activation activation;

   @Override
   public CostGradientTuple evaluate(Vector vector, WeightVector theta) {
      double predicted = activation.apply(theta.dot(vector));
      double y = vector.getLabelAsDouble();
      double derivative = lossFunction.derivative(predicted, y);
      return CostGradientTuple.of(lossFunction.loss(predicted, y),
                                  Gradient.of(vector.mapMultiply(derivative), derivative));
   }
}//END OF GradientDescentCostFunction
