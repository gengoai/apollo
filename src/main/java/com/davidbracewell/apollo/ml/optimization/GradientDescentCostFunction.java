package com.davidbracewell.apollo.ml.optimization;

/**
 * @author David B. Bracewell
 */

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.optimization.loss.LossFunction;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value
public class GradientDescentCostFunction implements CostFunction<LinearModelParameters> {
   LossFunction lossFunction;

   @Override
   public CostGradientTuple evaluate(NDArray vector, LinearModelParameters theta) {
      NDArray predicted = theta.activate(vector);
      NDArray y = vector.getLabelAsNDArray(theta.getNumberOfWeightVectors());
      NDArray derivative = lossFunction.derivative(predicted, y);
      return CostGradientTuple.of(lossFunction.loss(predicted, y), GradientParameter.calculate(vector, derivative));
   }
}//END OF GradientDescentCostFunction
