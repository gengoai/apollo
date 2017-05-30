package com.davidbracewell.apollo.optimization;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class GradientDescentUpdater implements WeightUpdater, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public void reset() {
   }

   @Override
   public void update(Weights weights, Weights gradient, double learningRate) {
      weights.getTheta().subtractSelf(gradient.getTheta().scale(learningRate));
      weights.getBias().subtractSelf(gradient.getBias().mapMultiply(learningRate));
   }


}// END OF GradientDescentUpdater
