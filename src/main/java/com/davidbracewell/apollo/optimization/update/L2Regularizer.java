package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.GradientMatrix;
import com.davidbracewell.apollo.optimization.WeightMatrix;

import java.io.Serializable;
import java.util.Iterator;

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
   public double update(WeightMatrix weights, GradientMatrix gradient, double learningRate, int iteration) {
      if (l2 == 0) {
         return super.update(weights, gradient, learningRate, iteration);
      }
      double cost = 0;
      for (int i = 0; i < weights.numberOfWeightVectors(); i++) {
         Vector w = weights.getWeightVector(i);
         double sum = 0;
         for (Iterator<Vector.Entry> itr = w.nonZeroIterator(); itr.hasNext(); ) {
            Vector.Entry e = itr.next();
            sum += (e.value * e.value);
         }
         cost += l2 * sum / 2d;
         gradient.get(i).getWeightGradient().mapMultiplySelf(l2 * sum / 2d);
      }
      super.update(weights, gradient, learningRate, iteration);
      return cost;
   }
}// END OF L2Regularizer
