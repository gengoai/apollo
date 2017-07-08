package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.optimization.Gradient;
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
   public double update(Weights weights, Gradient gradient, double learningRate) {
      super.update(weights, gradient, learningRate);
      if (l1 == 0) {
         return 0d;
      }

      double shrinkage = l1 * learningRate;
      AtomicDouble addedCost = new AtomicDouble();
      weights.getTheta().nonZeroIterator().forEachRemaining(entry -> {
         double nW = FastMath.signum(entry.value) * FastMath.max(0.0, FastMath.abs(entry.value) - shrinkage);
         if (FastMath.abs(nW) <= tolerance) {
            weights.getTheta().set(entry.row,entry.column,0d);
         }
         addedCost.addAndGet(FastMath.abs(entry.value));
         weights.getTheta().set(entry.row, entry.column, nW);
      });
//      weights.getTheta().mapSelf(weight -> {
//         double nW = FastMath.signum(weight) * FastMath.max(0.0, FastMath.abs(weight) - shrinkage);
//         if (FastMath.abs(nW) <= tolerance) {
//            return 0;
//         }
//         addedCost.addAndGet(FastMath.abs(weight));
//         return nW;
//      });
      return addedCost.get() * l1;
   }
}// END OF L1Regularizer
