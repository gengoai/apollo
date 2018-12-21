package com.gengoai.apollo.ml.sequence;

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.conversion.Cast;
import com.gengoai.stream.MStream;

import java.io.PrintStream;
import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class PerInstanceEvaluation implements Evaluation<SequenceLabeler>, Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The wrapped classifier evaluation
    */
   private final MultiClassEvaluation eval;


   public PerInstanceEvaluation() {
      this.eval = new MultiClassEvaluation();
   }

   @Override
   public void evaluate(SequenceLabeler model, MStream<Example> dataset) {
      dataset.forEach(sequence -> {
         Labeling result = model.label(sequence);
         for (int i = 0; i < sequence.size(); i++) {
            eval.entry(sequence.getExample(i).getLabelAsString(), result.getLabel(i));
         }
      });
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


   public void output(PrintStream printStream, boolean printConfusionMatrix) {
      eval.output(printStream, printConfusionMatrix);
   }

}//END OF PerInstanceEvaluation
