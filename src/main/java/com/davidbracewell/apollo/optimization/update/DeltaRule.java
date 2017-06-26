package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.optimization.Weights;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class DeltaRule implements WeightUpdate, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public double update(Weights weights, Weights gradient, double learningRate) {
      weights.getTheta().subtractSelf(gradient.getTheta().scale(learningRate));
      weights.getBias().subtractSelf(gradient.getBias().mapMultiply(learningRate));
      return 0;
   }


}// END OF DeltaRule
