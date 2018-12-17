package com.gengoai.apollo.ml.sequence;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.classification.Classification;
import com.gengoai.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface SequenceLabeler extends Model, Copyable<SequenceLabeler> {

   /**
    * Specialized transform to predict an outcome of the given NDArray, returning a {@link Classification} which is more
    * easily inheritable..
    *
    * @param data the NDArray input data
    */
   default Labeling label(NDArray data) {
      return new Labeling(estimate(data).getPredictedAsNDArray());
   }

   default SequenceLabeler copy() {
      return Copyable.copy(this);
   }

   @Override
   default Evaluation evaluate(MStream<NDArray> evaluationData) {
      return null;
   }
}//END OF SequenceLabeler
