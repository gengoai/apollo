package com.davidbracewell.apollo.optimization;

import com.davidbracewell.Math2;

/**
 * @author David B. Bracewell
 */
public class Sigmoid implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return Math2.clip(1. / (1. + Math.pow(Math.E, -x)), -30, 30);
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
