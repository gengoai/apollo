package com.davidbracewell.apollo.optimization.activation;

import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class TanH implements Activation {

   @Override
   public double apply(double x) {
      double ez = FastMath.exp(x);
      double enz = FastMath.exp(-x);
      return (ez - enz) / (ez + enz);
   }

   @Override
   public double valueGradient(double activated) {
      return 1.0 - (activated * activated);
   }

}// END OF TanH
