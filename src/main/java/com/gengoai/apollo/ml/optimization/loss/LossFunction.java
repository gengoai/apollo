package com.gengoai.apollo.ml.optimization.loss;

import com.gengoai.apollo.linear.NDArray;

/**
 * The interface Loss function.
 *
 * @author David B. Bracewell
 */
public interface LossFunction {

   /**
    * Derivative nd array.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the nd array
    */
   default NDArray derivative(NDArray predictedValue, NDArray trueValue) {
      return predictedValue.map(trueValue, this::derivative);
   }

   /**
    * Derivative double.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the double
    */
   double derivative(double predictedValue, double trueValue);

   /**
    * Loss double.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the double
    */
   default double loss(NDArray predictedValue, NDArray trueValue) {
      return predictedValue.map(trueValue, this::loss).scalarSum();
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
