package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Model;

/**
 * <p>Labels each example in a sequence of examples, which may represent points in time, tokens in a sentence, etc.
 * </p>
 *
 * @author David B. Bracewell
 */
public interface SequenceLabeler extends Model {

   /**
    * Specialized transform to predict the labels for a sequence.
    *
    * @param data the NDArray input data
    */
   default Labeling label(NDArray data) {
      return new Labeling(estimate(data).getPredictedAsNDArray());
   }


}//END OF SequenceLabeler
