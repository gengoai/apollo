package com.davidbracewell.apollo.optimization.regularization;

import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class L1Regularizer extends DeltaRule implements Serializable {
   private static final long serialVersionUID = 1L;

   private final double l1;
   private final double tolerance;

   public L1Regularizer(double l1) {
      this(l1, l1);
   }


   public L1Regularizer(double l1, double tolerance) {
      this.l1 = l1;
      this.tolerance = tolerance;
   }


   @Override
   public double update(Weights weights, Weights gradient, double learningRate) {
      super.update(weights, gradient, learningRate);
      if (l1 == 0) {
         return 0d;
      }

      double shrinkage = l1 * learningRate;
      AtomicDouble addedCost = new AtomicDouble();
      weights.getTheta().mapSelf(weight -> {
         double nW = FastMath.signum(weight) * FastMath.max(0.0, FastMath.abs(weight) - shrinkage);
         if (FastMath.abs(nW) <= tolerance) {
            return 0;
         }
         addedCost.addAndGet(FastMath.abs(weight));
         return nW;
      });
      return addedCost.get() * l1;
   }
}// END OF L1Regularizer
