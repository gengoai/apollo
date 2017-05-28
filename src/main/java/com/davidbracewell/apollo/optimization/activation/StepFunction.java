package com.davidbracewell.apollo.optimization.activation;

/**
 * @author David B. Bracewell
 */
public class StepFunction implements Activation {
   private static final long serialVersionUID = 1L;
   private final double threshold;

   public StepFunction() {
      this(0);
   }

   public StepFunction(double threshold) {
      this.threshold = threshold;
   }


   @Override
   public double apply(double x) {
      return x > threshold ? 1 : 0;
   }
}// END OF StepFunction
