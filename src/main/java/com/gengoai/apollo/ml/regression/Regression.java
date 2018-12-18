package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Model;

/**
 * <p>Base regression model that produces a real-value for an input instance.</p>
 *
 * @author David B. Bracewell
 */
public interface Regression extends Model {

   /**
    * Estimates a real-value based on the input instance.
    *
    * @param vector the instance
    * @return the estimated value
    */
   default double estimateScalar(NDArray vector) {
      return estimate(vector).scalarValue();
   }


   @Override
   default int getNumberOfLabels() {
      return 0;
   }
}//END OF Regression
