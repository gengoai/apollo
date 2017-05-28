package com.davidbracewell.apollo.optimization.activation;

/**
 * @author David B. Bracewell
 */
public class LinearFunction implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return x;
   }

}// END OF LinearActivation
