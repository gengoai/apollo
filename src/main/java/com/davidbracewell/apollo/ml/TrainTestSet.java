package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.Spliterator;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

/**
 * <p>Encapsulates a set of train/test splits.</p>
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
public class TrainTestSet<T extends Example> implements Iterable<TrainTestSplit<T>>, Serializable {
   private static final long serialVersionUID = 1L;
   private final Set<TrainTestSplit<T>> splits = new HashSet<>();

   /**
    * Adds a split to the set.
    *
    * @param trainTestSplit the train test split
    */
   public void add(@NonNull TrainTestSplit<T> trainTestSplit) {
      splits.add(trainTestSplit);
   }

   /**
    * Processes each split as a train/test pair
    *
    * @param consumer the consumer to run over the train & test splits
    */
   public void forEach(@NonNull BiConsumer<Dataset<T>, Dataset<T>> consumer) {
      forEach(tTrainTest -> consumer.accept(tTrainTest.getTrain(), tTrainTest.getTest()));
   }

   /**
    * <p>Evaluates the set using an Evaluation produced by the given supplier. The process begins by resetting the
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
      R eval = supplier.get();
      forEach(tt -> eval.merge(tt.evaluate(learner, supplier)));
      return eval;
   }

   @Override
   public Spliterator<TrainTestSplit<T>> spliterator() {
      return splits.spliterator();
   }

   @Override
   public Iterator<TrainTestSplit<T>> iterator() {
      return splits.iterator();
   }

   /**
    * Preprocess each of the training splits using the supplier of {@link PreprocessorList}
    *
    * @param supplier the supplier to produce a {@link PreprocessorList}
    * @return this train test set
    */
   public TrainTestSet<T> preprocess(@NonNull Supplier<PreprocessorList<T>> supplier) {
      forEach((train, test) -> train.preprocess(supplier.get()));
      return this;
   }


}// END OF TrainTestSet
