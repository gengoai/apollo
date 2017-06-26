package com.davidbracewell.apollo.optimization;

/**
 * @author David B. Bracewell
 */
public class BottouLearningRate implements LearningRate {

   private final double eta0;
   private final double lambda;

   public BottouLearningRate(double eta0, double lambda) {
      this.eta0 = eta0;
      this.lambda = lambda;
   }

   @Override
   public double get(double currentLearningRate, int time, int numProcessed) {
      return eta0 * 1.0 / (1 + eta0 * lambda * numProcessed);
   }

   @Override
   public double getInitialRate() {
      return eta0;
   }
}// END OF BottouLearningRate
