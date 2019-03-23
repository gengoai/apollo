package com.gengoai.apollo.optimization.loss;

import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

/**
 * <p>Squared loss</p>
 *
 * @author David B. Bracewell
 */
public class SquaredLoss implements LossFunction, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public double derivative(double predictedValue, double trueValue) {
      return 2.0 * (predictedValue - trueValue);
   }

   @Override
   public double loss(double predictedValue, double trueValue) {
      return FastMath.pow(trueValue - predictedValue, 2.0);
   }

}// END OF SquaredLoss
