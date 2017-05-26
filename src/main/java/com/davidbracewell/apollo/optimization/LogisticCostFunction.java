package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public class LogisticCostFunction implements StochasticCostFunction {
   public final DifferentiableLossFunction ll = LossFunctions.SQUARED;
   public final Activation activation = new Step(1);

//   public final DifferentiableLossFunction ll = LossFunctions.LOGISTIC;
//   public final Activation activation = new Sigmoid();
//
   public static double sigmoid(double x) {
      return 1. / (1. + Math.pow(Math.E, -x));
   }

   @Override
   public CostGradientTuple observe(Vector next, Vector weights) {
      double h = activation.apply(next.dot(weights));
      double y = next.getLabel();
      return new CostGradientTuple(ll.calculate(h, y), ll.gradient(next, h, y));
   }

}// END OF LogisticCostFunction
