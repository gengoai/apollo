/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml.data;

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
import java.util.List;
import java.util.Random;

import static com.gengoai.Validation.checkArgument;

/**
 * <p>A dataset is a collection of examples which can be used for training and evaluating models. Implementations of
 * dataset may store examples in memory, off heap, or distributed using Spark.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Dataset implements Iterable<Example>, Serializable, AutoCloseable {
   private static final long serialVersionUID = 1L;
   private final DatasetType datasetType;

   /**
    * Instantiates a new Dataset.
    *
    * @param datasetType the dataset type
    */
   protected Dataset(DatasetType datasetType) {
      this.datasetType = datasetType;
   }


   /**
    * Gets a {@link DatasetBuilder} so that a new {@link Dataset} can be created.
    *
    * @return the dataset builder
    */
   public static DatasetBuilder builder() {
      return new DatasetBuilder();
   }


   /**
    * Adds all the examples in the stream to the dataset.
    *
    * @param stream the stream
    */
   protected abstract void addAll(MStream<Example> stream);

   /**
    * Caches the examples in dataset.
    *
    * @return the cached dataset
    */
   public abstract Dataset cache();

   /**
    * Calculates the distribution of classes in the data set
    *
    * @return A counter containing the classes (labels) and their counts in the dataset
    */
   public Counter<String> calculateClassDistribution() {
      MCounterAccumulator<String> accumulator = getStreamingContext().counterAccumulator();
      stream().flatMap(Example::getStringLabelSpace).forEach(accumulator::add);
      return accumulator.value();
   }

   /**
    * Generates <code>numberOfFolds</code> {@link Split}s for cross-validation. Each split will have
    * <code>dataset.size() / numberOfFolds</code> testing data and the remaining data as training data.
    *
    * @param numberOfFolds the number of folds
    * @return An array of {@link Split} for each fold of the dataset
    */
   public Split[] fold(int numberOfFolds) {
      checkArgument(numberOfFolds > 0, "Number of folds must be >= 0");
      checkArgument(size() >= numberOfFolds, "Number of folds must be <= number of examples");
      Split[] folds = new Split[numberOfFolds];
      int foldSize = size() / numberOfFolds;
      for (int i = 0; i < numberOfFolds; i++) {
         int testStart = i * foldSize;
         int testEnd = testStart + foldSize;
         MStream<Example> testStream = stream(testStart, testEnd);
         MStream<Example> trainStream = getStreamingContext().empty();
         if (testStart > 0) {
            trainStream = trainStream.union(stream(0, testStart));
         }
         if (testEnd < size()) {
            trainStream = trainStream.union(stream(testEnd, size()));
         }
         folds[i] = new Split(newDataset(trainStream), newDataset(testStream));
      }
      return folds;
   }

   /**
    * Gets a streaming context compatible with this dataset
    *
    * @return the streaming context
    */
   public StreamingContext getStreamingContext() {
      return getType().getStreamingContext();
   }

   /**
    * Gets the type of this dataset
    *
    * @return the {@link DatasetType}
    */
   public final DatasetType getType() {
      return datasetType;
   }

   /**
    * Maps the examples in this dataset using the given function and creating a new dataset in the process.
    *
    * @param function the function to transform the examples
    * @return the dataset with the transformed examples
    */
   public Dataset map(SerializableFunction<? super Example, ? extends Example> function) {
      return newDataset(stream().map(function));
   }

   protected Dataset newDataset(MStream<Example> instances) {
      return getType().create(instances.map(Example::copy));
   }

   /**
    * Creates a balanced dataset by oversampling the items
    *
    * @return the balanced dataset
    */
   public Dataset oversample() {
      Counter<String> fCount = calculateClassDistribution();
      int targetCount = (int) fCount.maximumCount();

      Dataset dataset = newDataset(getStreamingContext().empty());

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
      return newDataset(stream().sample(withReplacement, sampleSize).map(e -> Cast.as(e.copy())));
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
   public Dataset shuffle(Random random) {
      return newDataset(stream().shuffle(random));
   }

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
      return newDataset(stream().skip(start).limit(end - start));
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
   public abstract MStream<Example> stream();

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
      Dataset dataset = newDataset(getStreamingContext().empty());
      for (Object label : fCount.items()) {
         dataset.addAll(stream().filter(e -> e.getStringLabelSpace().anyMatch(label::equals))
                                .sample(false, targetCount));
      }
      return dataset;
   }

   /**
    * Writes the dataset to the given location using one json per line.
    *
    * @param location the location to write the dataset
    * @throws IOException Something went wrong writing
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
