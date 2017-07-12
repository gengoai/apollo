package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.GradientMatrix;
import com.davidbracewell.apollo.optimization.WeightMatrix;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class L2Regularizer extends DeltaRule implements Serializable {
   private static final long serialVersionUID = 1L;

   private final double l2;

   public L2Regularizer(double l2) {
      this.l2 = l2;
   }

   @Override
   public double update(WeightMatrix weights, GradientMatrix gradient, double learningRate) {
      if (l2 == 0) {
         return super.update(weights, gradient, learningRate);
      }
      double cost = 0;
      for (int i = 0; i < weights.numberOfWeightVectors(); i++) {
         Vector w = weights.getWeightVector(i);
         cost += l2 * w.map(d -> d * d).sum() / 2d;
         gradient.get(i).getWeightGradient().addSelf(w.mapMultiply(l2));
      }
      super.update(weights, gradient, learningRate);
      return cost;
   }
}// END OF L2Regularizer
