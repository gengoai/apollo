package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import lombok.NonNull;
import lombok.Value;

import java.io.Serializable;
import java.util.function.Supplier;

/**
 * The type Train test.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
@Value
public class TrainTest<T extends Example> implements Serializable {
   private static final long serialVersionUID = 1L;
   private Dataset<T> train;
   private Dataset<T> test;

   /**
    * Of train test.
    *
    * @param <T>   the type parameter
    * @param train the train
    * @param test  the test
    * @return the train test
    */
   public static <T extends Example> TrainTest<T> of(@NonNull Dataset<T> train, @NonNull Dataset<T> test) {
      return new TrainTest<>(train, test);
   }

   public <M extends Model, R extends Evaluation<T, M>> R evaluate(Learner<T, M> learner, Supplier<R> supplier) {
      learner.reset();
      R eval = supplier.get();
      eval.evaluate(learner.train(train), test);
      return eval;
   }

}// END OF TrainTest
