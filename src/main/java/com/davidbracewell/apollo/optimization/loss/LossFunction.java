package com.davidbracewell.apollo.optimization.loss;

import com.davidbracewell.apollo.linalg.SinglePointVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.CostGradientTuple;
import lombok.NonNull;

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
    * Derivative double.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the double
    */
   default Vector derivative(@NonNull Vector predictedValue, @NonNull Vector trueValue) {
      return predictedValue.map(trueValue, this::derivative);
   }

   /**
    * Loss double.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the double
    */
   double loss(double predictedValue, double trueValue);

   /**
    * Loss double.
    *
    * @param predictedValue the predicted value
    * @param trueValue      the true value
    * @return the double
    */
   default double loss(@NonNull Vector predictedValue, @NonNull Vector trueValue) {
      return predictedValue.map(trueValue, this::loss).sum();
   }


   default CostGradientTuple lossAndDerivative(@NonNull Vector predictedValue, @NonNull Vector trueValue) {
      return new CostGradientTuple(loss(predictedValue, trueValue), derivative(predictedValue, trueValue));
   }

   default CostGradientTuple lossAndDerivative(double predictedValue, double trueValue) {
      return new CostGradientTuple(loss(predictedValue, trueValue),
                                   new SinglePointVector(derivative(predictedValue, trueValue)));
   }

}//END OF LossFunction
