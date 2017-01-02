package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import lombok.NonNull;
import lombok.Value;

import java.io.Serializable;
import java.util.function.Supplier;

/**
 * <p>Encapsulates a single training and test split using two datasets.</p>
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
@Value
public class TrainTestSplit<T extends Example> implements Serializable {
   private static final long serialVersionUID = 1L;
   private Dataset<T> train;
   private Dataset<T> test;

   /**
    * Convenience method for creating a train test split
    *
    * @param <T>   the example type parameter
    * @param train the training dataset
    * @param test  the testing dataset
    * @return the train test split
    */
   public static <T extends Example> TrainTestSplit<T> of(@NonNull Dataset<T> train, @NonNull Dataset<T> test) {
      return new TrainTestSplit<>(train, test);
   }

   /**
    * <p>Evaluates the split using an Evaluation produced by the given supplier. The process begins by resetting the
    * learner and the trains the learner using the training portion of the split. Once the model is built it is
    * evaluated on the testing portion of the split.</p>
    *
    * @param <M>      the model type parameter
    * @param <R>      the evaluation type parameter
    * @param learner  the learner to use for training
    * @param supplier supplies an evaluation metric.
    * @return the result of evaluation
    */
   public <M extends Model, R extends Evaluation<T, M>> R evaluate(@NonNull Learner<T, M> learner, @NonNull Supplier<R> supplier) {
      learner.reset();
      R eval = supplier.get();
      eval.evaluate(learner.train(train), test);
      return eval;
   }

}// END OF TrainTest
