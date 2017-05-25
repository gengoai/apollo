package com.davidbracewell.apollo.optimization;

/**
 * @author David B. Bracewell
 */
public class LinearActivation implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return x;
   }

   @Override
   public double gradient(double x) {
      return 1d;
   }

   @Override
   public double valueGradient(double x) {
      return 1d;
   }
}// END OF LinearActivation
