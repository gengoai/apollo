package com.davidbracewell.apollo.optimization.activation;

/**
 * @author David B. Bracewell
 */
public class ReLuActivation implements Activation {

   @Override
   public double apply(double x) {
      return Math.max(0, x);
   }

   @Override
   public double valueGradient(double activated) {
      return activated > 0 ? 1 : 0;
   }


}//END OF ReLuActivation
