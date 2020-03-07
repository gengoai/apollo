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

import com.gengoai.annotation.JsonHandler;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Pipeline;
import com.gengoai.apollo.ml.Split;
import com.gengoai.collection.counter.Counter;
import com.gengoai.function.SerializableFunction;
import com.gengoai.io.resource.Resource;
import com.gengoai.json.Json;
import com.gengoai.json.JsonEntry;
import com.gengoai.stream.StreamingContext;
import com.gengoai.stream.MCounterAccumulator;
import lombok.NonNull;

import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.List;
import java.util.Random;

import static com.gengoai.Validation.notNull;

/**
 * <p>A dataset is a collection of examples which can be used for training and evaluating models. Implementations of
 * dataset may store examples in memory, off heap, or distributed using Spark.</p>
 *
 * @author David B. Bracewell
 */
@JsonHandler(ExampleDataset.JsonMarshaller.class)
public abstract class ExampleDataset implements Dataset<Example, ExampleDataset> {
   private static final long serialVersionUID = 1L;

   /**
    * Gets a {@link DatasetBuilder} so that a new {@link ExampleDataset} can be created.
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
   public VectorizedDataset toVectorizedDataset(final Pipeline pipeline) {
      return toVectorizedDataset(getType(), pipeline);
   }

   /**
    * Uses the given pipeline to create a stream of {@link NDArray} from the dataset
    *
    * @param datasetType the type of dataset to create
    * @param pipeline    the pipeline
    * @return the stream of NDArray
    */
   public VectorizedDataset toVectorizedDataset(@NonNull final DatasetType datasetType,
                                                @NonNull final Pipeline pipeline) {
      notNull(pipeline, "Pipeline must not be null");
      return datasetType.createVectorizedDataset(stream().map(e -> e.transform(pipeline)));
   }

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
    * Maps the examples in this dataset using the given function and creating a new dataset in the process.
    *
    * @param function the function to transform the examples
    * @return the dataset with the transformed examples
    */
   public abstract ExampleDataset map(SerializableFunction<? super Example, ? extends Example> function);

   /**
    * Creates a balanced dataset by oversampling the items
    *
    * @return the balanced dataset
    */
   public abstract ExampleDataset oversample();

   /**
    * Samples the dataset creating a new dataset of the given sample size.
    *
    * @param withReplacement the with replacement
    * @param sampleSize      the sample size
    * @return the dataset
    */
   public abstract ExampleDataset sample(boolean withReplacement, int sampleSize);

   @Override
   public final ExampleDataset shuffle() {
      return shuffle(new Random(0));
   }

   @Override
   public abstract ExampleDataset shuffle(Random random);

   @Override
   public long size() {
      return stream().count();
   }

   @Override
   public abstract ExampleDataset slice(long start, long end);

   /**
    * Split the dataset into a train and test split.
    *
    * @param pctTrain the percentage of the dataset to use for training
    * @return A TestTrainSet of one TestTrain item
    */
   public abstract Split split(double pctTrain);


   @Override
   public List<Example> take(int n) {
      return stream().take(n);
   }

   /**
    * Creates a balanced dataset by undersampling the items
    *
    * @return the balanced dataset
    */
   public abstract ExampleDataset undersample();

   /**
    * Writes the dataset to the given location using one json per line.
    *
    * @param location the location to write the dataset
    * @throws IOException Something went wrong writing
    */
   public void write(Resource location) throws IOException {
      try(BufferedWriter writer = new BufferedWriter(location.writer())) {
         for(Example example : this) {
            writer.write(Json.dumps(example));
            writer.write("\n");
         }
      }
   }

   public static class JsonMarshaller extends com.gengoai.json.JsonMarshaller<ExampleDataset> {
      @Override
      protected ExampleDataset deserialize(JsonEntry entry, Type type) {
         return ExampleDataset.builder()
                              .type(entry.getProperty("type").getAs(DatasetType.class))
                              .source(StreamingContext.local()
                                                      .stream(entry.getProperty("examples").getAsArray(Example.class)));
      }

      @Override
      protected JsonEntry serialize(ExampleDataset examples, Type type) {
         return JsonEntry.object()
                         .addProperty("type", examples.getType())
                         .addProperty("examples", JsonEntry.array(examples));
      }
   }

}//END OF Dataset
