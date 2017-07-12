package com.davidbracewell.apollo.optimization.loss;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * The interface Loss function.
 *
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


   /**
    * Derivative vector.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the vector
    */
   default Vector derivative(Vector predictedValue, Vector trueValue) {
      return predictedValue.map(trueValue, this::derivative);
   }

   /**
    * Loss double.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the double
    */
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
