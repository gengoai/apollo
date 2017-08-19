package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;

/**
 * @author David B. Bracewell
 */
public class ReLuActivation implements Activation {

   @Override
   public double apply(double x) {
      return Math.max(0, x);
   }


   @Override
   public Matrix apply(Matrix m) {
      return m.max(0);
   }

   @Override
   public Matrix valueGradient(Matrix m) {
      return m.predicate(x -> x > 0);
   }

   @Override
   public double valueGradient(double activated) {
      return activated > 0 ? 1 : 0;
   }
}//END OF ReLuActivation
