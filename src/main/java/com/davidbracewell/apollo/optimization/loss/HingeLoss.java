package com.davidbracewell.apollo.optimization.loss;

import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class HingeLoss implements LossFunction, Serializable {
   private static final long serialVersionUID = 1L;
   private final double threshold;

   public HingeLoss() {
      this(1);
   }

   public HingeLoss(double threshold) {
      this.threshold = threshold;
   }

   @Override
   public double derivative(double predictedValue, double trueValue) {
      trueValue = trueValue == 0 ? -1 : 1;
      predictedValue = predictedValue == 0 ? -1 : 1;
      if (trueValue * predictedValue > threshold) {
         return 0;
      }
      return -trueValue;
   }

   @Override
   public double loss(double predictedValue, double trueValue) {
      trueValue = trueValue == 0 ? -1 : 1;
      predictedValue = predictedValue == 0 ? -1 : 1;
      return FastMath.max(0, threshold - trueValue * predictedValue);
   }
}// END OF HingeLoss
