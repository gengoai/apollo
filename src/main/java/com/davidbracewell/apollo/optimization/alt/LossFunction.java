package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public interface LossFunction {

   /**
    * Derivative double.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the double
    */
   double derivative(double predictedValue, double trueValue);


   default Vector derivative(Vector predictedValue, Vector trueValue) {
      return predictedValue.map(trueValue, this::derivative);
   }

   default double loss(Vector predictedValue, Vector trueValue) {
      return predictedValue.map(trueValue, this::loss).sum();
   }

   /**
    * Loss double.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the double
    */
   double loss(double predictedValue, double trueValue);

}//END OF LossFunction
