package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.val;

/**
 * @author David B. Bracewell
 */
public class SoftmaxActivation implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return Activation.SIGMOID.apply(x);
   }

   @Override
   public Vector apply(Vector x) {
      double max = x.max();
      x.mapSelf(d -> Math.exp(d - max));
      double sum = x.sum();
      return x.mapDivideSelf(sum);
   }


   @Override
   public Matrix apply(Matrix m) {
      val max = m.columnMaxs();
      val exp = m.subRowVector(max).exp();
      val sums = exp.columnSums();
      return exp.diviRowVector(sums);
   }

   @Override
   public boolean isProbabilistic() {
      return true;
   }

   @Override
   public Vector valueGradient(Vector activated) {
      Vector gradient = Vector.dZeros(activated.dimension());
      for (int i = 0; i < activated.dimension(); i++) {
         double vi = activated.get(i);
         double sum = 0;
         for (int j = 0; j < activated.dimension(); j++) {
            if (i == j) {
               sum += vi * (1 - vi);
            } else {
               sum += -vi * activated.get(j);
            }
         }
         gradient.set(i, sum);
      }
      return gradient;
   }


   @Override
   public double valueGradient(double activated) {
      return activated * (1d - activated);
   }

   @Override
   public Matrix valueGradient(Matrix m) {
      return m.mul(m.rsub(1.0));
   }

}// END OF SoftmaxActivation
