package com.davidbracewell.apollo.optimization;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class ConstantLearningRate implements LearningRate, Serializable {
   private static final long serialVersionUID = 1L;

   private final double learningRate;

   public ConstantLearningRate(double learningRate) {
      this.learningRate = learningRate;
   }

   @Override
   public double get(double currentLearningRate, int time, int numProcessed) {
      return learningRate;
   }

   @Override
   public double getInitialRate() {
      return learningRate;
   }

}//END OF ConstantLearningRate
