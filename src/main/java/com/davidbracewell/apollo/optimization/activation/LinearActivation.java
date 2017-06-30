package com.davidbracewell.apollo.optimization.activation;

/**
 * The type Linear function.
 *
 * @author David B. Bracewell
 */
public class LinearActivation implements DifferentiableActivation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return x;
   }

   @Override
   public double valueGradient(double activated) {
      return 1;
   }

}// END OF LinearActivation
