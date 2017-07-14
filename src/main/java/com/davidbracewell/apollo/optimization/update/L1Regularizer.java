package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.GradientMatrix;
import com.davidbracewell.apollo.optimization.WeightMatrix;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

import static com.davidbracewell.collection.list.Lists.asArrayList;

/**
 * @author David B. Bracewell
 */
public class L1Regularizer extends DeltaRule implements Serializable {
   private static final long serialVersionUID = 1L;

   private final double l1;
   private final double tolerance;

   public L1Regularizer(double l1) {
      this(l1, 0.000001);
   }


   public L1Regularizer(double l1, double tolerance) {
      this.l1 = l1;
      this.tolerance = tolerance;
   }


   @Override
   public double update(WeightMatrix weights, GradientMatrix gradient, double learningRate) {
      super.update(weights, gradient, learningRate);
      if (l1 == 0) {
         return 0d;
      }
      double shrinkage = l1 * learningRate;
      double addedCost = 0d;
      for (int wi = 0; wi < weights.numberOfWeightVectors(); wi++) {
         Vector w = weights.getWeightVector(wi);
         for (Vector.Entry entry : asArrayList(w.nonZeroIterator())) {
            double nW = FastMath.signum(entry.value) * FastMath.max(0.0, FastMath.abs(entry.value) - shrinkage);
            if (tolerance != 0 && FastMath.abs(nW) <= tolerance) {
               w.set(entry.index, 0d);
            }
            addedCost += FastMath.abs(entry.value);
            w.set(entry.index, nW);
         }
      }
      return addedCost * l1;
   }
}// END OF L1Regularizer
