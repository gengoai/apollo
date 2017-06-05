package com.davidbracewell.apollo.optimization.activation;

/**
 * @author David B. Bracewell
 */
public class ReLuActivation implements Activation {

   @Override
   public double apply(double x) {
      return Math.max(0, x);
   }

}//END OF ReLuActivation
