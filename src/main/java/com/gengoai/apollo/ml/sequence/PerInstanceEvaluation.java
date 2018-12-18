package com.gengoai.apollo.ml.sequence;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;

import java.io.PrintStream;
import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class PerInstanceEvaluation implements Evaluation, Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The wrapped classifier evaluation
    */
   private final MultiClassEvaluation eval;


   public PerInstanceEvaluation(int numberOfLabels) {
      this.eval = new MultiClassEvaluation(numberOfLabels);
   }

   public PerInstanceEvaluation(Vectorizer<String> vectorizer) {
      this.eval = new MultiClassEvaluation(vectorizer);
   }


   @Override
   public void entry(NDArray entry) {
      eval.entry(entry);
   }

   @Override
   public void merge(Evaluation evaluation) {
      Validation.checkArgument(evaluation instanceof PerInstanceEvaluation);
      eval.merge(Cast.<PerInstanceEvaluation>as(evaluation).eval);
   }

   @Override
   public void output(PrintStream printStream) {
      eval.output(printStream, true);
   }

}//END OF PerInstanceEvaluation
