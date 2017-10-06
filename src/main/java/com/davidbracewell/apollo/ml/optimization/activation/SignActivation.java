package com.davidbracewell.apollo.ml.optimization.activation;

import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class SignActivation implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return FastMath.signum(x);
   }

   @Override
   public double valueGradient(double activated) {
      return 2 * activated;
   }


}// END OF SignActivation

