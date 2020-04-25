/*
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
 */

package com.gengoai.apollo.ml;

import com.gengoai.annotation.JsonHandler;
import com.gengoai.apollo.ml.observation.Variable;
import com.gengoai.apollo.ml.transform.Transform;
import com.gengoai.collection.counter.Counter;
import com.gengoai.function.SerializableFunction;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonMarshaller;
import com.gengoai.stream.MCounterAccumulator;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.tuple.Tuples;
import lombok.NonNull;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.*;
import java.util.function.Consumer;

/**
 * <p>
 * A dataset is a collection of {@link Datum} and is used to represent the training, testing, and development data for
 * machine learning models. Each dataset keeps track of metadata for the {@link com.gengoai.apollo.ml.observation.Observation}s
 * in its datum which define the dimension (number of possible values), type (Sequence, Variable, etc.), and any
 * associated {@link com.gengoai.apollo.ml.encoder.Encoder}. It is the responsibility of individual {@link Transform} to
 * ensure that the metadata is kept updated.
 * </p>
 * <p>
 * Note: Many of the machine learning algorithms rely on the metadata to determine the dimensions of the input and
 * output variables. Thus, it is important that if you are defining a dataset where the sources are NDArray that you
 * manually set the metadata on the Dataset. For example:
 * </p>
 * <pre>
 * {@code
 *      List<Datum> data = Arrays.asList(
 *         Datum.of($("input", <NDARRAY>),
 *                  $("output", <NDARRAY>))
 *                  ...
 *      );
 *      DataSet dataset = new InMemoryDataset(data);
 *      dataset.updateMetadata("input", m -> m.setDimension(100));
 *      dataset.updateMetadata("output", m -> m.setDimension(20));
 * }
 * </pre>
 *
 * @author David B. Bracewell
 */
@JsonHandler(DataSet.Marshaller.class)
public abstract class DataSet implements Iterable<Datum>, Serializable {
   private static final long serialVersionUID = 1L;
   protected final Map<String, ObservationMetadata> metadata = new HashMap<>();

   /**
    * Generates an iterator of "batches" by partitioning the datum into groups of given batch size.
    *
    * @param batchSize the batch size
    * @return the iterator
    */
   public abstract Iterator<DataSet> batchIterator(int batchSize);

   /**
    * Caches the dataset into memory.
    *
    * @return the cached dataset
    */
   public abstract DataSet cache();

   /**
    * Calculates the distribution of classes in the data set
    *
    * @param observationName The name of the observation to calculate the distribution over
    * @return A counter containing the classes (labels) and their counts in the dataset
    */
   public Counter<String> calculateClassDistribution(@NonNull String observationName) {
      MCounterAccumulator<String> accumulator = getStreamingContext().counterAccumulator();
      stream().flatMap(e -> e.get(observationName)
                             .getVariableSpace()
                             .map(Variable::getName)).forEach(accumulator::add);
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
    * Gets the metadata for the observation sources on the datum in this dataset describing the dimension, type, and any
    * associated encoder.
    *
    * @param source the source
    * @return the metadata
    */
   public ObservationMetadata getMetadata(@NonNull String source) {
      return metadata.get(source);
   }

   /**
    * Gets the map of source name - {@link ObservationMetadata} for this dataset
    *
    * @return the map of source name - {@link ObservationMetadata}
    */
   public Map<String, ObservationMetadata> getMetadata() {
      return metadata;
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
    * Gets the type of this DataSet
    *
    * @return the DataSetType
    */
   public abstract DataSetType getType();

   /**
    * Maps the datum in the dataset constructing a new dataset. Depending on the underlying implementation this method
    * may be performed lazily.
    *
    * @param function the function to apply to  the datum in the dataset
    * @return this dataset
    */
   public abstract DataSet map(SerializableFunction<? super Datum, ? extends Datum> function);

   /**
    * Creates a balanced dataset by oversampling the items
    *
    * @param observationName The name of the observation to oversample over
    * @return the balanced dataset
    */
   public abstract DataSet oversample(@NonNull String observationName);

   /**
    * Generates a parallel MStream over the datum in this dataset
    *
    * @return parallel stream of data in the dataset
    */
   public abstract MStream<Datum> parallelStream();

   /**
    * Probes the data set to determine the types of its observations. This is only necessary if the metadata is needed
    * directly after constructing a dataset.
    *
    * @return this DataSet
    */
   public DataSet probe() {
      parallelStream()
            .flatMap(d -> d.entrySet().stream())
            .map(e -> Tuples.$(e.getKey(), e.getValue().getClass()))
            .distinct()
            .forEach(e -> updateMetadata(e.getKey(), m -> m.setType(e.getValue())));
      return this;
   }

   /**
    * Copies all metadata from the given map to this data set.
    *
    * @param metadata the metadata
    * @return this DataSet
    */
   public DataSet putAllMetadata(@NonNull Map<String, ObservationMetadata> metadata) {
      this.metadata.putAll(metadata);
      return this;
   }

   /**
    * Removes the metadata associated with a given observation source.
    *
    * @param source the observation source
    * @return this DataSet
    */
   public DataSet removeMetadata(@NonNull String source) {
      metadata.remove(source);
      return this;
   }

   /**
    * Samples the dataset creating a new dataset of the given sample size.
    *
    * @param withReplacement the with replacement
    * @param sampleSize      the sample size
    * @return the dataset
    */
   public abstract DataSet sample(boolean withReplacement, int sampleSize);

   /**
    * Shuffles the data in the dataset.
    *
    * @return This dataset with its data shuffled
    */
   public DataSet shuffle() {
      return shuffle(new Random());
   }

   /**
    * Shuffles the dataset creating a new one with the given random number generator.
    *
    * @param random the random number generator
    * @return the dataset
    */
   public abstract DataSet shuffle(Random random);

   /**
    * Returns the number of datum in the dataset
    *
    * @return The number of datum in the dataset
    */
   public abstract long size();

   /**
    * Creates a new dataset containing instances from the given <code>start</code> index upto the given <code>end</code>
    * index.
    *
    * @param start the starting item index (Inclusive)
    * @param end   the ending item index (Exclusive)
    * @return the dataset
    */
   public abstract DataSet slice(long start, long end);

   /**
    * Split the dataset into a train and test split.
    *
    * @param pctTrain the percentage of the dataset to use for training
    * @return A TestTrainSet of one TestTrain item
    */
   public abstract Split split(double pctTrain);

   /**
    * Generates an MStream over the datum in this dataset
    *
    * @return stream of data in the dataset
    */
   public abstract MStream<Datum> stream();

   /**
    * Takes the first n elements from the dataset
    *
    * @param n the number of items to take
    * @return the list of items
    */
   public List<Datum> take(int n) {
      return stream().take(n);
   }

   /**
    * Creates a balanced dataset by undersampling the items
    *
    * @param observationName The name of the observation to undersample over
    * @return the balanced dataset
    */
   public abstract DataSet undersample(@NonNull String observationName);

   /**
    * Updates the metadata associated with a given observation source.
    *
    * @param source  the observation source
    * @param updater the update consumer
    * @return this DataSet
    */
   public DataSet updateMetadata(@NonNull String source, @NonNull Consumer<ObservationMetadata> updater) {
      metadata.putIfAbsent(source, new ObservationMetadata());
      updater.accept(metadata.get(source));
      return this;
   }

   public static class Marshaller extends JsonMarshaller<DataSet> {
      private static final long serialVersionUID = 1L;

      @Override
      protected DataSet deserialize(JsonEntry entry, Type type) {
         List<Datum> data = new ArrayList<>();
         entry.getProperty("data")
              .elementIterator()
              .forEachRemaining(e -> {
                 System.out.println(e);
                 data.add(e.getAs(Datum.class));
              });
         DataSet dataSet = new InMemoryDataSet(data);
//         dataSet.putAllMetadata(entry.getProperty("metadata").getAsMap(ObservationMetadata.class));
         return dataSet;
      }

      @Override
      protected JsonEntry serialize(DataSet data, Type type) {
         JsonEntry obj = JsonEntry.object();
         obj.addProperty("metadata", data.getMetadata());
         JsonEntry array = JsonEntry.array();
         for(Datum datum : data) {
            array.addValue(datum);
         }
         obj.addProperty("data", array);
         return obj;
      }
   }

}//END OF DataSet
