package com.davidbracewell.apollo.optimization;

import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class GradientDescentUpdater implements WeightUpdater {

   @Getter
   @Setter
   private double learningRate = 1;

   private double eta;

   @Override
   public void reset() {
      eta = 0;
   }

   @Override
   public void update(Weights weights, Weights gradient) {
      if (eta == 0) {
         eta = learningRate;
      }
      weights.getTheta().subtractSelf(gradient.getTheta().scale(eta));
      weights.getBias().subtractSelf(gradient.getBias().mapMultiply(eta));
   }




}// END OF GradientDescentUpdater
