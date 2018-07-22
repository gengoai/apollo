package com.gengoai.apollo.ml.optimization.loss;

import com.gengoai.math.Math2;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class LogLoss implements LossFunction, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public double derivative(double predictedValue, double trueValue) {
      return predictedValue - trueValue;
   }

   @Override
   public double loss(double predictedValue, double trueValue) {
      predictedValue = Math2.clip(predictedValue, 1e-15, 1 - 1e-15);
      if (trueValue == 1) {
         return -FastMath.log(predictedValue);
      }
      return -FastMath.log(1.0 - predictedValue);
   }

}// END OF LogLoss
