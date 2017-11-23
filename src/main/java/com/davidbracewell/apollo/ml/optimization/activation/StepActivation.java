package com.davidbracewell.apollo.ml.optimization.activation;

import com.davidbracewell.apollo.linear.NDArray;

/**
 * The type Step function.
 *
 * @author David B. Bracewell
 */
public class StepActivation implements Activation {
   private static final long serialVersionUID = 1L;
   private final double threshold;

   /**
    * Instantiates a new Step function.
    */
   public StepActivation() {
      this(0);
   }

   /**
    * Instantiates a new Step function.
    *
    * @param threshold the threshold
    */
   public StepActivation(double threshold) {
      this.threshold = threshold;
   }


   @Override
   public double apply(double x) {
      return x > threshold ? 1 : 0;
   }


   @Override
   public double valueGradient(double activated) {
      return activated;
   }

   @Override
   public NDArray valueGradient(NDArray activated) {
      return activated;
   }

}// END OF StepActivation
