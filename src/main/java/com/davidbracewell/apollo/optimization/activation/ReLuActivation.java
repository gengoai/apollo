package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class ReLuActivation implements DifferentiableActivation {

   @Override
   public double apply(double x) {
      return Math.max(0, x);
   }

   @Override
   public Vector valueGradient(Vector activated) {
      return activated.map(x -> x > 0 ? 1 : 0);
   }

   @Override
   public Vector valueGradient(@NonNull Vector predicted, @NonNull Vector actual) {
      return predicted.map(x -> x > 0 ? 1 : 0)
                      .multiplySelf(actual);
   }

   @Override
   public Matrix valueGradient(@NonNull Matrix predicted, @NonNull Matrix actual) {
      return predicted.map(x -> x > 0 ? 1 : 0)
                      .ebeMultiplySelf(actual);
   }

}//END OF ReLuActivation
