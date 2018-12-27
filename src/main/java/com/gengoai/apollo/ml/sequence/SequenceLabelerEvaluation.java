package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface SequenceLabelerEvaluation extends Evaluation {
   /**
    * Evaluate the given model using the given dataset
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   default void evaluate(SequenceLabeler model, Dataset dataset) {
      evaluate(model, dataset.stream());
   }

   /**
    * Evaluate the given model using the given set of examples
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   void evaluate(SequenceLabeler model, MStream<Example> dataset);
}//END OF SequenceLabelerEvaluation
