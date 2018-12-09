package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.classification.Classification;

/**
 * @author David B. Bracewell
 */
public interface SequenceLabeler extends Model {

   /**
    * Specialized transform to predict an outcome of the given NDArray, returning a {@link Classification} which is more
    * easily inheritable..
    *
    * @param data the NDArray input data
    */
   default Labeling label(NDArray data) {
      return new Labeling(estimate(data).getPredictedAsNDArray());
   }

}//END OF SequenceLabeler
