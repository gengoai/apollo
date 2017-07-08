package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;
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
      this(l1, l1);
   }


   public L1Regularizer(double l1, double tolerance) {
      this.l1 = l1;
      this.tolerance = tolerance;
   }


   @Override
   public double update(WeightVector weights, Gradient gradient, double learningRate) {
      super.update(weights, gradient, learningRate);
      if (l1 == 0) {
         return 0d;
      }
      double shrinkage = l1 * learningRate;
      double addedCost = 0d;
      for (Vector.Entry entry : asArrayList(weights.getWeights().nonZeroIterator())) {
         double nW = FastMath.signum(entry.value) * FastMath.max(0.0, FastMath.abs(entry.value) - shrinkage);
         if (FastMath.abs(nW) <= tolerance) {
            weights.getWeights().set(entry.index, 0d);
         }
         addedCost += FastMath.abs(entry.value);
         weights.getWeights().set(entry.index, nW);
      }
      return addedCost * l1;
   }
}// END OF L1Regularizer
