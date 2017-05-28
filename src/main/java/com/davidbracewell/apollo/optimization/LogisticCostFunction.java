package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.loss.HingeLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;

/**
 * @author David B. Bracewell
 */
public class LogisticCostFunction implements StochasticCostFunction {
   public final LossFunction ll = new HingeLoss();
   public final Activation activation = new Step(1);

   //   public final DifferentiableLossFunction ll = LossFunctions.LOGISTIC;
//   public final Activation activation = new Sigmoid();
//
   public static double sigmoid(double x) {
      return 1. / (1. + Math.pow(Math.E, -x));
   }

   @Override
   public CostGradientTuple observe(Vector next, Weights weights) {
      Vector h = weights.dot(next).map(activation::apply);
      Vector y = next.getLabelVector(weights.numClasses());
      return ll.lossAndDerivative(h, y);
   }

}// END OF LogisticCostFunction
