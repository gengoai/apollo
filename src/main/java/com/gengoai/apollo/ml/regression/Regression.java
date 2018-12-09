package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Model;

/**
 * @author David B. Bracewell
 */
public interface Regression extends Model {

   default double estimateScalar(NDArray vector) {
      return estimate(vector).scalarValue();
   }

}//END OF Regression
