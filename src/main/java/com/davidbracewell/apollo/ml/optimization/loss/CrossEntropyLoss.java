package com.davidbracewell.apollo.ml.optimization.loss;

import com.davidbracewell.Math2;
import com.davidbracewell.apollo.linear.NDArray;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class CrossEntropyLoss implements LossFunction, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public double derivative(double predictedValue, double trueValue) {
      return predictedValue - trueValue;
   }

   @Override
   public double loss(NDArray predictedValue, NDArray trueValue) {
      return -trueValue.map(predictedValue, (d1, d2) -> d1 * Math2.safeLog(d2)).sum();
   }

   @Override
   public double loss(double predictedValue, double trueValue) {
      return -(trueValue * Math2.safeLog(predictedValue));
   }


}// END OF CrossEntropyLoss
