package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.classification.ClassifierEvaluation;
import com.davidbracewell.conversion.Cast;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.PrintStream;
import java.util.Collection;

/**
 * @author David B. Bracewell
 */
public class PerInstanceEvaluation implements Evaluation<Sequence, SequenceLabeler> {
  ClassifierEvaluation eval = new ClassifierEvaluation();

  @Override
  public void evaluate(@NonNull SequenceLabeler model, @NonNull Dataset<Sequence> dataset) {
    dataset.forEach(sequence -> {
      LabelingResult result = model.label(sequence);
      for (int i = 0; i < sequence.size(); i++) {
        eval.entry(sequence.get(i).getLabel().toString(), result.getLabel(i));
      }
    });
  }

  @Override
  public void evaluate(@NonNull SequenceLabeler model, @NonNull Collection<Sequence> dataset) {
    dataset.forEach(sequence -> {
      LabelingResult result = model.label(sequence);
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
    eval.output(printStream);
  }

}// END OF PerInstanceEvaluation
