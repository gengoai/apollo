package com.davidbracewell.apollo.optimization.o2;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.CostGradientTuple;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.loss.LossFunction;

/**
 * @author David B. Bracewell
 */
public class GradientDescentCostFunction implements CostFunction {
   private final LossFunction lossFunction;
   private final Activation activation;

   public GradientDescentCostFunction(LossFunction lossFunction, Activation activation) {
      this.lossFunction = lossFunction;
      this.activation = activation;
   }

   @Override
   public CostGradientTuple evaluate(Vector vector, WeightComponent theta) {
      Vector predicted = activation.apply(theta.get(0).dot(vector));
      Vector y = vector.getLabelVector(predicted.dimension());
      return lossFunction.lossAndDerivative(predicted, y);
   }

}// END OF SGDLinearCostFunction
