package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

/**
 * @author David B. Bracewell
 */
public class TrainTestSet<T extends Example> extends ArrayList<TrainTest<T>> {

  public void forEach(@NonNull BiConsumer<Dataset<T>, Dataset<T>> consumer) {
    forEach(tTrainTest -> consumer.accept(tTrainTest.getTrain(), tTrainTest.getTest()));
  }

  public <M extends Model, R extends Evaluation<T, M>> R evaluate(Learner<T, M> learner, Supplier<R> supplier) {
    R eval = supplier.get();
    forEach(tt -> eval.merge(tt.evaluate(learner, supplier)));
    return eval;
  }

  public TrainTestSet<T> preprocess(Supplier<PreprocessorList<T>> supplier) {
    forEach((train, test) ->
      train.preprocess(supplier.get())
    );
    return this;
  }


}// END OF TrainTestSet
