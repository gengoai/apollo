package com.gengoai.apollo.ml.optimization.activation;


import com.gengoai.apollo.linear.NDArray;

/**
 * @author David B. Bracewell
 */
public class ReLuActivation implements Activation {

   @Override
   public double apply(double x) {
      return Math.max(0, x);
   }


   @Override
   public NDArray valueGradient(NDArray activated) {
      return activated.test(d -> d > 0);
   }

   @Override
   public double valueGradient(double activated) {
      return activated > 0 ? 1 : 0;
   }
}//END OF ReLuActivation
