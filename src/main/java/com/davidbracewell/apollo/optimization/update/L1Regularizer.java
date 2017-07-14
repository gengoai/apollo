package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.GradientMatrix;
import com.davidbracewell.apollo.optimization.WeightMatrix;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

import static com.davidbracewell.collection.list.Lists.asArrayList;

/**
 * The type L 1 regularizer.
 *
 * @author David B. Bracewell
 */
public class L1Regularizer extends DeltaRule implements Serializable {
   private static final long serialVersionUID = 1L;

   private final double l1;
   private final double tolerance;

   /**
    * Instantiates a new L 1 regularizer.
    */
   public L1Regularizer() {
      this(1);
   }

   /**
    * Instantiates a new L 1 regularizer.
    *
    * @param l1 the l 1
    */
   public L1Regularizer(double l1) {
      this(l1, 1e-9);
   }


   /**
    * Instantiates a new L 1 regularizer.
    *
    * @param l1        the l 1
    * @param tolerance the tolerance
    */
   public L1Regularizer(double l1, double tolerance) {
      this.l1 = l1;
      this.tolerance = tolerance;
   }


   @Override
   public double update(WeightMatrix weights, GradientMatrix gradient, double learningRate, int iteration) {
      super.update(weights, gradient, learningRate, iteration);
      if (l1 == 0) {
         return 0d;
      }
      double shrinkage = l1 * (learningRate / FastMath.sqrt(iteration));
      double addedCost = 0d;
      for (int wi = 0; wi < weights.numberOfWeightVectors(); wi++) {
         Vector w = weights.getWeightVector(wi);
         for (Vector.Entry entry : asArrayList(w.nonZeroIterator())) {
            double abs = FastMath.abs(entry.value);
            double nW = FastMath.signum(entry.value) * FastMath.max(0.0, abs - shrinkage);
            if (FastMath.abs(nW) < tolerance) {
               w.set(entry.index, 0d);
            } else {
               w.set(entry.index, nW);
            }
            addedCost += abs;
         }
      }
      return addedCost * l1;
   }
}// END OF L1Regularizer
