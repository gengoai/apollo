package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class TanH implements DifferentiableActivation {

   @Override
   public double apply(double x) {
      double ez = FastMath.exp(x);
      double enz = FastMath.exp(-x);
      return (ez - enz) / (ez + enz);
   }

   @Override
   public Vector valueGradient(Vector activated) {
      return activated.map(x -> 1.0 - (x * x));
   }

   @Override
   public Vector valueGradient(@NonNull Vector predicted, @NonNull Vector actual) {
      return predicted.map(x -> 1.0 - (x * x))
                      .multiplySelf(actual);
   }

   @Override
   public Matrix valueGradient(@NonNull Matrix predicted, @NonNull Matrix actual) {
      return predicted.map(x -> 1.0 - (x * x))
                      .ebeMultiplySelf(actual);
   }

}// END OF TanH
