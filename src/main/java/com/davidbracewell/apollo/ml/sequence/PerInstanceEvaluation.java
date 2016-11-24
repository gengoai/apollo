package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.classification.ClassifierEvaluation;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.conversion.Cast;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.PrintStream;
import java.util.Collection;

/**
 * Sequence evaluation that calculates metrics using each element of the sequence wrapping a {@link
 * ClassifierEvaluation}
 *
 * @author David B. Bracewell
 */
public class PerInstanceEvaluation implements Evaluation<Sequence, SequenceLabeler> {
   /**
    * The wrapped classifier evaluation
    */
   public final ClassifierEvaluation eval = new ClassifierEvaluation();

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
