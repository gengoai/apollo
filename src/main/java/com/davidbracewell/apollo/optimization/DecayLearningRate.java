package com.davidbracewell.apollo.optimization;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class DecayLearningRate implements LearningRate, Serializable {
   private static final long serialVersionUID = 1L;

   private final double decayRate;
   private final double initialRate;


   public DecayLearningRate(double initialRate, double decayRate) {
      this.decayRate = decayRate;
      this.initialRate = initialRate;
   }

   @Override
   public double get(double currentLearningRate, int time, int numProcessed) {
      return currentLearningRate * 1.0 / (1.0 + decayRate * time);
   }

   @Override
   public double getInitialRate() {
      return initialRate;
   }
}// END OF ExponentialDecay
