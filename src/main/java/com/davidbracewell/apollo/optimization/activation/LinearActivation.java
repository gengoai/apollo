package com.davidbracewell.apollo.optimization.activation;

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


}// END OF LinearActivation
