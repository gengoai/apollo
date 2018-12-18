package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Model;

/**
 * Base class for classifiers that predicts the label, or class, for a set of features.
 *
 * @author David B. Bracewell
 */
public interface Classifier extends Model {


   /**
    * Specialized transform to predict an outcome of the given NDArray, returning a {@link Classification} which is more
    * easily manipulated.
    *
    * @param data the NDArray input data
    * @return the classification result
    */
   default Classification predict(NDArray data) {
      return new Classification(estimate(data).getPredictedAsNDArray());
   }


}//END OF Classifier
