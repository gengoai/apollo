package com.davidbracewell.apollo.optimization;

/**
 * @author David B. Bracewell
 */
public class Sigmoid implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      if( x > 0 ) {
         return  1.0 / (1.0 + Math.exp(-x));
      }
      double z = Math.exp(x);
      return z / (1 + z);
   }

   @Override
   public double gradient(double x) {
      return valueGradient(apply(x));
   }

   @Override
   public double valueGradient(double x) {
      return x * (1 - x);
   }
}// END OF Sigmoid
