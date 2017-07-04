package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
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
      CostGradientTuple tuple = lossFunction.lossAndDerivative(predicted, y);
      return CostGradientTuple.of(tuple.getLoss(), tuple.getGradient().respectToInput(vector));
   }

}// END OF SGDLinearCostFunction
