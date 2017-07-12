package com.davidbracewell.apollo.optimization.alt.again;


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
