package com.davidbracewell.apollo.optimization.loss;

import com.davidbracewell.Math2;
import com.davidbracewell.apollo.linalg.Vector;

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
   public double loss(double predictedValue, double trueValue) {
      return -(trueValue * Math2.safeLog(predictedValue));
   }

   @Override
   public double loss(Vector predictedValue, Vector trueValue) {
      double loss = 0;
      for (int i = 0; i < predictedValue.dimension(); i++) {
         loss += loss(predictedValue.get(i), trueValue.get(i));
      }
      return loss;
   }

}// END OF CrossEntropyLoss
