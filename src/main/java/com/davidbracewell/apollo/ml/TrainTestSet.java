package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

/**
 * The type Train test set.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class TrainTestSet<T extends Example> extends ArrayList<TrainTest<T>> {
  private static final long serialVersionUID = 1L;

  /**
   * For each.
   *
   * @param consumer the consumer
   */
  public void forEach(@NonNull BiConsumer<Dataset<T>, Dataset<T>> consumer) {
    forEach(tTrainTest -> consumer.accept(tTrainTest.getTrain(), tTrainTest.getTest()));
  }

  /**
   * Evaluate r.
   *
   * @param <M>      the type parameter
   * @param <R>      the type parameter
   * @param learner  the learner
   * @param supplier the supplier
   * @return the r
   */
  public <M extends Model, R extends Evaluation<T, M>> R evaluate(@NonNull Learner<T, M> learner, @NonNull Supplier<R> supplier) {
    R eval = supplier.get();
    forEach(tt -> eval.merge(tt.evaluate(learner, supplier)));
    return eval;
  }

  /**
   * Preprocess train test set.
   *
   * @param supplier the supplier
   * @return the train test set
   */
  public TrainTestSet<T> preprocess(@NonNull Supplier<PreprocessorList<T>> supplier) {
    forEach((train, test) ->
      train.preprocess(supplier.get())
    );
    return this;
  }


}// END OF TrainTestSet
