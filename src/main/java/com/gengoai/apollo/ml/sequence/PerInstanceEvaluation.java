package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.mango.conversion.Cast;
import com.gengoai.guava.common.base.Preconditions;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.data.Dataset;
import lombok.NonNull;

import java.io.PrintStream;
import java.util.Collection;

/**
 * Sequence evaluation that calculates metrics using each element of the sequence wrapping a {@link
 * MultiClassEvaluation}
 *
 * @author David B. Bracewell
 */
public class PerInstanceEvaluation implements Evaluation<Sequence, SequenceLabeler> {
   /**
    * The wrapped classifier evaluation
    */
   public final MultiClassEvaluation eval = new MultiClassEvaluation();

   @Override
   public void evaluate(@NonNull SequenceLabeler model, @NonNull Dataset<Sequence> dataset) {
      dataset.forEach(sequence -> {
         Labeling result = model.label(sequence);
         for (int i = 0; i < sequence.size(); i++) {
            eval.entry(sequence.get(i).getLabel().toString(), result.getLabel(i));
         }
      });
   }

   @Override
   public void evaluate(@NonNull SequenceLabeler model, @NonNull Collection<Sequence> dataset) {
      dataset.forEach(sequence -> {
         Labeling result = model.label(sequence);
         for (int i = 0; i < sequence.size(); i++) {
            eval.entry(sequence.get(i).getLabel().toString(), result.getLabel(i));
         }
      });
   }

   @Override
   public void merge(@NonNull Evaluation<Sequence, SequenceLabeler> evaluation) {
      Preconditions.checkArgument(evaluation instanceof PerInstanceEvaluation);
      eval.merge(Cast.<PerInstanceEvaluation>as(evaluation).eval);
   }

   @Override
   public void output(@NonNull PrintStream printStream) {
      eval.output(printStream, false);
   }

   /**
    * Outputs the results of the classification as per-class Precision, Recall, and F1 and optionally the confusion
    * matrix.
    *
    * @param printStream          the print stream to write to
    * @param printConfusionMatrix True print the confusion matrix, False do not print the confusion matrix.
    */
   public void output(@NonNull PrintStream printStream, boolean printConfusionMatrix) {
      eval.output(printStream, printConfusionMatrix);
   }


}// END OF PerInstanceEvaluation
