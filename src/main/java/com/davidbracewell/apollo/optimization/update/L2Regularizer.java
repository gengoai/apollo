package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.Gradient;
import com.davidbracewell.apollo.optimization.Weights;

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
   public double update(Weights weights, Gradient gradient, double learningRate) {
      if (l2 == 0) {
         return super.update(weights, gradient, learningRate);
      }
      double cost = 0;
      for (int r = 0; r < weights.getTheta().numberOfRows(); r++) {
         Vector row = weights.getTheta().row(r);
         cost += l2 * row.map(d -> d * d).sum() / 2d;
         gradient.getWeightGradient().row(r).addSelf(row.mapMultiply(l2));
      }
      super.update(weights, gradient, learningRate);
      return cost;
   }
}// END OF L2Regularizer
