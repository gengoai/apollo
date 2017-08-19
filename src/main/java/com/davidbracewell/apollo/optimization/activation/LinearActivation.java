package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;

/**
 * The type Linear function.
 *
 * @author David B. Bracewell
 */
public class LinearActivation implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return x;
   }

   @Override
   public double valueGradient(double activated) {
      return 1;
   }

   @Override
   public Matrix valueGradient(Matrix m) {
      return null;
   }

   @Override
   public Matrix apply(Matrix m) {
      return m;
   }

}// END OF LinearActivation
