package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class SigmoidActivation implements DifferentiableActivation {
   public static final SigmoidActivation INSTANCE = new SigmoidActivation();
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      if (x > 0) {
         return 1.0 / (1.0 + Math.exp(-x));
      }
      double z = Math.exp(x);
      return z / (1 + z);
   }

   @Override
   public boolean isProbabilistic() {
      return true;
   }

   @Override
   public Vector valueGradient(Vector activated) {
      return activated.map(d -> d * (1.0 - d));
   }

   @Override
   public Matrix valueGradient(@NonNull Matrix predicted, @NonNull Matrix actual) {
      return predicted.map(d -> d * (1.0 - d))
                      .ebeMultiplySelf(actual);
   }

   @Override
   public Vector valueGradient(@NonNull Vector predicted, @NonNull Vector actual) {
      return predicted.map(d -> d * (1.0 - d))
                      .multiplySelf(actual);
   }
}// END OF SigmoidActivation
