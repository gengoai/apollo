package com.davidbracewell.apollo.optimization;

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
   public double gradient(double x) {
      return 0;
   }

   @Override
   public double valueGradient(double x) {
      return 0;
   }
}// END OF SignActivation
