package com.gengoai.apollo.ml.data;

import com.gengoai.Copyable;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Split;
import com.gengoai.collection.counter.Counter;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableFunction;
import com.gengoai.io.resource.Resource;
import com.gengoai.json.Json;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.stream.accumulator.MCounterAccumulator;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static com.gengoai.Validation.checkArgument;

/**
 * <p>A dataset is a collection of examples which can be used for training and evaluating models. Implementations of
 * dataset may store examples in memory, off heap, or distributed using Spark.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Dataset implements Iterable<Example>, Copyable<Dataset>, Serializable, AutoCloseable {
   private static final long serialVersionUID = 1L;

   /**
    * Adds all the examples in the stream to the dataset.
    *
    * @param stream the stream
    */
   protected abstract void addAll(MStream<Example> stream);

   /**
    * Add all the examples in the collection to the dataset
    *
    * @param instances the instances
    */
   protected void addAll(Collection<Example> instances) {
      addAll(getType().getStreamingContext().stream(instances));
   }

   /**
    * Calculate class distribution counter.
    *
    * @return the counter
    */
   public Counter<String> calculateClassDistribution() {
      MCounterAccumulator<String> accumulator = getStreamingContext().counterAccumulator();
      stream().flatMap(Example::getStringLabelSpace).forEach(accumulator::add);
      return accumulator.value();
   }

   /**
    * Creates folds for cross-validation
    *
    * @param numberOfFolds the number of folds
    * @return the TrainTestSet made of the number of folds
    */
   public Split[] fold(int numberOfFolds) {
      checkArgument(numberOfFolds > 0, "Number of folds must be >= 0");
      checkArgument(size() >= numberOfFolds, "Number of folds must be <= number of examples");
      Split[] folds = new Split[numberOfFolds];
      int foldSize = size() / numberOfFolds;
      for (int i = 0; i < numberOfFolds; i++) {
         Dataset train = newSimilarDataset(getStreamingContext().empty());
         Dataset test = newSimilarDataset(getStreamingContext().empty());

         int testStart = i * foldSize;
         int testEnd = testStart + foldSize;

         test.addAll(stream(testStart, testEnd));

         if (testStart > 0) {
            train.addAll(stream(0, testStart));
         }

         if (testEnd < size()) {
            train.addAll(stream(testEnd, size()));
         }

         folds[i] = new Split(train, test);
      }
      return folds;
   }

   /**
    * Gets streaming context.
    *
    * @return the streaming context
    */
   public StreamingContext getStreamingContext() {
      return getType().getStreamingContext();
   }

   /**
    * Gets type.
    *
    * @return the type
    */
   public abstract DatasetType getType();

   /**
    * Map dataset.
    *
    * @param function the function
    * @return the dataset
    */
   public Dataset map(SerializableFunction<? super Example, ? extends Example> function) {
      return newSimilarDataset(stream().map(function));
   }

   /**
    * Applies the given function modifying the instances of this dataset.
    *
    * @param function The function to apply to the examples
    * @return This dataset
    */
   public abstract Dataset mapSelf(SerializableFunction<? super Example, ? extends Example> function);

   /**
    * New similar dataset dataset.
    *
    * @param instances the instances
    * @return the dataset
    */
   protected abstract Dataset newSimilarDataset(MStream<Example> instances);

   /**
    * Creates a balanced dataset by oversampling the items
    *
    * @return the balanced dataset
    */
   public Dataset oversample() {
      Counter<String> fCount = calculateClassDistribution();
      int targetCount = (int) fCount.maximumCount();

      Dataset dataset = newSimilarDataset(getStreamingContext().empty());

      for (Object label : fCount.items()) {
         MStream<Example> fStream = stream()
                                       .filter(e -> e.getStringLabelSpace().anyMatch(label::equals))
                                       .cache();
         int count = (int) fStream.count();
         int curCount = 0;
         while (curCount + count < targetCount) {
            dataset.addAll(fStream);
            curCount += count;
         }
         if (curCount < targetCount) {
            dataset.addAll(fStream.sample(false, targetCount - curCount));
         } else if (count == targetCount) {
            dataset.addAll(fStream);
         }
      }
      return dataset;
   }

   /**
    * Samples the dataset creating a new dataset of the given sample size.
    *
    * @param withReplacement the with replacement
    * @param sampleSize      the sample size
    * @return the dataset
    */
   public Dataset sample(boolean withReplacement, int sampleSize) {
      checkArgument(sampleSize > 0, "Sample size must be > 0");
      return newSimilarDataset(stream().sample(withReplacement, sampleSize).map(e -> Cast.as(e.copy())));
   }

   /**
    * Shuffles the dataset creating a new dataset.
    *
    * @return the dataset
    */
   public final Dataset shuffle() {
      return shuffle(new Random(0));
   }

   /**
    * Shuffles the dataset creating a new one with the given random number generator.
    *
    * @param random the random number generator
    * @return the dataset
    */
   public abstract Dataset shuffle(Random random);

   /**
    * The number of examples in the dataset
    *
    * @return the number of examples
    */
   public int size() {
      return (int) stream().count();
   }

   /**
    * Creates a new dataset containing instances from the given <code>start</code> index upto the given <code>end</code>
    * index.
    *
    * @param start the starting item index (Inclusive)
    * @param end   the ending item index (Exclusive)
    * @return the dataset
    */
   public Dataset slice(int start, int end) {
      return newSimilarDataset(stream().skip(start).limit(end - start));
   }

   /**
    * Split the dataset into a train and test split.
    *
    * @param pctTrain the percentage of the dataset to use for training
    * @return A TestTrainSet of one TestTrain item
    */
   public Split split(double pctTrain) {
      checkArgument(pctTrain > 0 && pctTrain < 1, "Percentage should be between 0 and 1");
      int split = (int) Math.floor(pctTrain * size());
      return new Split(slice(0, split), slice(split, size()));
   }

   /**
    * Creates an MStream of examples from this Dataset.
    *
    * @return the MStream of examples
    */
   public MStream<Example> stream() {
      return StreamingContext.local().stream(this);
   }

   /**
    * Slices the dataset into a sub stream
    *
    * @param start the starting item index (Inclusive)
    * @param end   the ending item index (Exclusive)
    * @return the stream
    */
   protected MStream<Example> stream(int start, int end) {
      return stream().skip(start).limit(end - start).cache();
   }

   /**
    * Takes the first n elements from the dataset
    *
    * @param n the number of items to take
    * @return the list of items
    */
   public List<Example> take(int n) {
      return stream().take(n);
   }

   /**
    * Creates a balanced dataset by undersampling the items
    *
    * @return the balanced dataset
    */
   public Dataset undersample() {
      Counter<String> fCount = calculateClassDistribution();
      int targetCount = (int) fCount.minimumCount();
      Dataset dataset = newSimilarDataset(getStreamingContext().empty());
      for (Object label : fCount.items()) {
         dataset.addAll(stream().filter(e -> e.getStringLabelSpace().anyMatch(label::equals))
                                .sample(false, targetCount));
      }
      return dataset;
   }

   /**
    * Write.
    *
    * @param location the location
    * @throws IOException the io exception
    */
   public void write(Resource location) throws IOException {
      try (BufferedWriter writer = new BufferedWriter(location.writer())) {
         for (Example example : this) {
            writer.write(Json.dumps(example));
            writer.write("\n");


         }
      }
   }


}//END OF Dataset
