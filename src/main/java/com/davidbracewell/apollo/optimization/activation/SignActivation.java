package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class SignActivation implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public Matrix apply(Matrix m) {
      return m.map(this::apply);
   }

   @Override
   public double apply(double x) {
      return FastMath.signum(x);
   }


   @Override
   public Matrix valueGradient(Matrix m) {
      return m.mul(2);
   }

   @Override
   public double valueGradient(double activated) {
      return 2 * activated;
   }


}// END OF SignActivation

