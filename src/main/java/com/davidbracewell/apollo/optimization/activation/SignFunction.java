package com.davidbracewell.apollo.optimization.activation;

import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class SignFunction implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return FastMath.signum(x);
   }

}// END OF SignFunction

