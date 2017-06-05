package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;

/**
 * @author David B. Bracewell
 */
public class LogisticCostFunction implements StochasticCostFunction {
   public final LossFunction loss = new LogLoss();
   public final Activation activation = new SigmoidActivation();

   @Override
   public LossGradientTuple observe(Vector next, Weights weights) {
      return loss.lossAndDerivative(activation.apply(weights.dot(next)), next.getLabelVector(weights.numClasses()));
   }

}// END OF LogisticCostFunction
