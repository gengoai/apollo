package com.davidbracewell.apollo.optimization.alt.again;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.alt.LossFunction;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value
public class GradientDescentCostFunction implements CostFunction {
   LossFunction lossFunction;
   Activation activation;

   @Override
   public CostGradientTuple evaluate(Vector vector, WeightMatrix theta) {
      Vector predicted = theta.isBinary()
                         ? theta.binaryDot(vector, activation)
                         : theta.dot(vector, activation);
      Vector y = vector.getLabelVector(theta.numberOfWeightVectors());
      Vector derivative = lossFunction.derivative(predicted, y);
      return CostGradientTuple.of(lossFunction.loss(predicted, y), GradientMatrix.calculate(vector, derivative));
   }
}//END OF GradientDescentCostFunction
