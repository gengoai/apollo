package com.davidbracewell.apollo.optimization.update;


import com.davidbracewell.apollo.optimization.GradientMatrix;
import com.davidbracewell.apollo.optimization.WeightMatrix;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class DeltaRule implements WeightUpdate, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public double update(WeightMatrix weights, GradientMatrix gradient, double learningRate) {
      weights.subtract(gradient);
      return 0;
   }


}// END OF DeltaRule
