package com.davidbracewell.apollo.optimization;

/**
 * @author David B. Bracewell
 */
public class Step implements Activation {
   private static final long serialVersionUID = 1L;
   private final double threshold;

   public Step() {
      this(0);
   }

   public Step(double threshold) {
      this.threshold = threshold;
   }


   @Override
   public double apply(double x) {
      return x > threshold ? 1 : 0;
   }

   @Override
   public double gradient(double x) {
      return 0;
   }

   @Override
   public double valueGradient(double x) {
      return 0;
   }
}// END OF Step
