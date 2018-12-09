package com.gengoai.apollo.ml.classification;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Model;

/**
 * The interface Classifier.
 *
 * @author David B. Bracewell
 */
public interface Classifier extends Model, Copyable<Classifier> {


   /**
    * Specialized transform to predict an outcome of the given NDArray, returning a {@link Classification} which is more
    * easily inheritable..
    *
    * @param data the NDArray input data
    * @return the classification result
    */
   default Classification predict(NDArray data) {
      return new Classification(estimate(data).getPredictedAsNDArray());
   }


}//END OF Classifier
