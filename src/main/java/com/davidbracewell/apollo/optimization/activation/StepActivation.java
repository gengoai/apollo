package com.davidbracewell.apollo.optimization.activation;

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


}// END OF StepActivation
