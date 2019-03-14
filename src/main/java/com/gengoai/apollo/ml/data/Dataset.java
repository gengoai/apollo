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

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Pipeline;
import com.gengoai.apollo.ml.Split;
import com.gengoai.collection.counter.Counter;
import com.gengoai.function.SerializableFunction;
import com.gengoai.io.resource.Resource;
import com.gengoai.json.Json;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.stream.accumulator.MCounterAccumulator;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import static com.gengoai.Validation.notNull;

/**
 * <p>A dataset is a collection of examples which can be used for training and evaluating models. Implementations of
 * dataset may store examples in memory, off heap, or distributed using Spark.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Dataset implements Iterable<Example>, Serializable, AutoCloseable {
   private static final long serialVersionUID = 1L;


   /**
    * Gets a {@link DatasetBuilder} so that a new {@link Dataset} can be created.
    *
    * @return the dataset builder
    */
   public static DatasetBuilder builder() {
      return new DatasetBuilder();
   }

   /**
    * Uses the given pipeline to create a stream of {@link NDArray} from the dataset
    *
    * @param pipeline the pipeline
    * @return the stream of NDArray
    */
   public MStream<NDArray> asVectorStream(final Pipeline pipeline) {
      notNull(pipeline, "Pipeline must not be null");
      return stream().map(e -> e.transform(pipeline));
   }

   /**
    * <p>Iterator that provides a batch of examples per iteration.</p>
    *
    * @param batchSize the batch size
    * @return the iterator
    */
   public abstract Iterator<Dataset> batchIterator(final int batchSize);

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
      stream().flatMap(Example::getLabelSpace).forEach(accumulator::add);
      return accumulator.value();
   }

   /**
    * Generates <code>numberOfFolds</code> {@link Split}s for cross-validation. Each split will have
    * <code>dataset.size() / numberOfFolds</code> testing data and the remaining data as training data.
    *
    * @param numberOfFolds the number of folds
    * @return An array of {@link Split} for each fold of the dataset
    */
   public abstract Split[] fold(int numberOfFolds);

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
   public abstract DatasetType getType();

   /**
    * Maps the examples in this dataset using the given function and creating a new dataset in the process.
    *
    * @param function the function to transform the examples
    * @return the dataset with the transformed examples
    */
   public abstract Dataset map(SerializableFunction<? super Example, ? extends Example> function);

   /**
    * Creates a balanced dataset by oversampling the items
    *
    * @return the balanced dataset
    */
   public Dataset oversample() {
//      Counter<String> fCount = calculateClassDistribution();
//      int targetCount = (int) fCount.maximumCount();
//
//      Dataset dataset = newDataset(getStreamingContext().empty());
//
//      for (Object label : fCount.items()) {
//         MStream<Example> fStream = stream()
//                                       .filter(e -> e.getLabelSpace().anyMatch(label::equals))
//                                       .cache();
//         int count = (int) fStream.count();
//         int curCount = 0;
//         while (curCount + count < targetCount) {
////            dataset.addAll(fStream);
//            curCount += count;
//         }
//         if (curCount < targetCount) {
////            dataset.addAll(fStream.sample(false, targetCount - curCount));
//         } else if (count == targetCount) {
////            dataset.addAll(fStream);
//         }
//      }
//      return dataset;
      return this;
   }

   /**
    * Samples the dataset creating a new dataset of the given sample size.
    *
    * @param withReplacement the with replacement
    * @param sampleSize      the sample size
    * @return the dataset
    */
   public abstract Dataset sample(boolean withReplacement, int sampleSize);

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
   public long size() {
      return stream().count();
   }

   /**
    * Creates a new dataset containing instances from the given <code>start</code> index upto the given <code>end</code>
    * index.
    *
    * @param start the starting item index (Inclusive)
    * @param end   the ending item index (Exclusive)
    * @return the dataset
    */
   public abstract Dataset slice(long start, long end);

   /**
    * Split the dataset into a train and test split.
    *
    * @param pctTrain the percentage of the dataset to use for training
    * @return A TestTrainSet of one TestTrain item
    */
   public abstract Split split(double pctTrain);

   /**
    * Creates an MStream of examples from this Dataset.
    *
    * @return the MStream of examples
    */
   public abstract MStream<Example> stream();

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
//      Counter<String> fCount = calculateClassDistribution();
//      int targetCount = (int) fCount.minimumCount();
//      Dataset dataset = newDataset(getStreamingContext().empty());
//      for (Object label : fCount.items()) {
////         dataset.addAll(stream().filter(e -> e.getLabelSpace().anyMatch(label::equals))
////                                .sample(false, targetCount));
//      }
//      return dataset;
      return this;
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
