package com.davidbracewell.apollo.optimization.alt;


import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class DeltaRule implements WeightUpdate, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public double update(WeightVector weights, Gradient gradient, double learningRate) {
      weights.update(gradient);
      return 0;
   }


}// END OF DeltaRule
