package com.davidbracewell.apollo.optimization.activation;

/**
 * @author David B. Bracewell
 */
public class ReLu implements Activation {

   @Override
   public double apply(double x) {
      return Math.max(0, x);
   }

}//END OF ReLu
