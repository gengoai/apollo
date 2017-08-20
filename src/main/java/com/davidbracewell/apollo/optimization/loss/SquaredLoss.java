package com.davidbracewell.apollo.optimization.loss;

import com.davidbracewell.apollo.linalg.Matrix;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

/**
 * The type Squared loss.
 *
 * @author David B. Bracewell
 */
public class SquaredLoss implements LossFunction, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public Matrix derivative(Matrix predictedValue, Matrix trueValue) {
      return predictedValue.sub(trueValue).mul(2);
   }

   @Override
   public double derivative(double predictedValue, double trueValue) {
      return 2.0 * (predictedValue - trueValue);
   }

   @Override
   public double loss(Matrix predictedValue, Matrix trueValue) {
      return trueValue.sub(predictedValue).mapi(x -> x * x).sum();
   }

   @Override
   public double loss(double predictedValue, double trueValue) {
      return FastMath.pow(trueValue - predictedValue, 2.0);
   }
}// END OF SquaredLoss
