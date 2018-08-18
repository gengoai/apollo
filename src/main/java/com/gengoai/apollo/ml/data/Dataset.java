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
 */

package com.gengoai.apollo.ml.data;

import com.gengoai.Copyable;
import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.TrainTestSet;
import com.gengoai.apollo.ml.TrainTestSplit;
import com.gengoai.apollo.ml.encoder.*;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.sequence.Sequence;
import com.gengoai.collection.counter.Counter;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableFunction;
import com.gengoai.function.SerializablePredicate;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.io.resource.Resource;
import com.gengoai.json.JsonReader;
import com.gengoai.json.JsonWriter;
import com.gengoai.logging.Logger;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.stream.accumulator.MCounterAccumulator;
import com.google.gson.stream.JsonToken;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Stream;

/**
 * <p>A dataset is a collection of examples which can be used for training and evaluating models. Implementations of
 * dataset may store examples in memory, off heap, or distributed using Spark.</p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public abstract class Dataset<T extends Example> implements Iterable<T>, Copyable<Dataset>, Serializable, AutoCloseable {
   private static final Logger log = Logger.getLogger(Dataset.class);
   private static final long serialVersionUID = 1L;
   private final EncoderPair encoders;
   private final PreprocessorList<T> preprocessors;

   /**
    * Instantiates a new Dataset.
    *
    * @param featureEncoder the feature encoder to use on the dataset
    * @param labelEncoder   the label encoder to use on the dataset
    * @param preprocessors  the preprocessors applied to the dataset
    */
   protected Dataset(Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      this.encoders = new EncoderPair(labelEncoder, featureEncoder);
      this.preprocessors = preprocessors == null ? PreprocessorList.empty() : preprocessors;
   }

   /**
    * Creates a dataset builder with an <code>IndexEncoder</code> for the class labels as is required for classification
    * problems.
    *
    * @return the dataset builder
    */
   public static DatasetBuilder<Instance> classification() {
      return new DatasetBuilder<>(new LabelIndexEncoder(), Instance.class);
   }

   /**
    * Creates a dataset for word embedding tasks.
    *
    * @param <T>       the component type of the streaming to create the dataset from
    * @param type      the dataset type to create
    * @param stream    the stream to convert into tokens
    * @param tokenizer function to use convert items of type <code>T</code> to <code>String</code>
    * @return the embedding dataset
    */
   public static <T> Dataset<Sequence> embedding(@NonNull DatasetType type, @NonNull MStream<T> stream, @NonNull SerializableFunction<T, Stream<String>> tokenizer) {
      return new DatasetBuilder<>(new NoOptLabelEncoder(), Sequence.class)
                .featureEncoder(new NoOptEncoder())
                .type(type)
                .source(stream.map(line -> Sequence.create(tokenizer.apply(line))));
   }

   /**
    * Creates a dataset builder with a <code>RealEncoder</code> for the class labels as is required for regression
    * problems.
    *
    * @return the dataset builder
    */
   public static DatasetBuilder<Instance> regression() {
      return new DatasetBuilder<>(new RegressionLabelEncoder(), Instance.class);
   }

   /**
    * Creates a dataset builder with an <code>IndexEncoder</code> for the class labels as is required for classification
    * problems.
    *
    * @return the dataset builder
    */
   public static DatasetBuilder<Sequence> sequence() {
      return new DatasetBuilder<>(new LabelIndexEncoder(), Sequence.class);
   }

   /**
    * Adds all the examples in the stream to the dataset.
    *
    * @param stream the stream
    */
   protected abstract void addAll(MStream<T> stream);

   /**
    * Add all the examples in the collection to the dataset
    *
    * @param instances the instances
    */
   protected void addAll(@NonNull Collection<T> instances) {
      addAll(getType().getStreamingContext().stream(instances));
   }

   /**
    * Add all the examples to the dataset
    *
    * @param instances the instances
    */
   @SafeVarargs
   protected final void addAll(@NonNull T... instances) {
      addAll(Arrays.asList(instances));
   }

   /**
    * Creates a stream of {@link com.gengoai.apollo.linear.NDArray} from the examples in the dataset
    *
    * @return the stream of FeatureVectors
    */
   public MStream<NDArray> asVectors() {
      encode();
      return stream().parallel()
                     .flatMap(ii -> ii.asInstances().stream())
                     .map(ii -> ii.toVector(encoders));
   }

   /**
    * Creates a stream of {@link NDArray} from the examples in the dataset with the label set to <code>true (1.0)</code>
    * when the actual label matches the given <code>trueLabel</code>
    *
    * @param trueLabel The label to match for a binary true
    * @return The stream of vectors
    */
   public MStream<NDArray> asVectors(double trueLabel) {
      encode();
      return stream().parallel()
                     .flatMap(ii -> ii.asInstances().stream())
                     .map(ii -> {
                        NDArray v = ii.toVector(encoders);
                        if (v.getLabelAsDouble() == trueLabel) {
                           v.setLabel(1d);
                        } else {
                           v.setLabel(0d);
                        }
                        return v;
                     });
   }

   /**
    * Caches the computations performed on the dataset.
    *
    * @return A cached version of the dataset
    */
   public Dataset<T> cache() {
      return this;
   }

   @Override
   public void close() throws Exception {

   }

   @Override
   public Dataset<T> copy() {
      return create(stream().map(e -> Cast.as(e.copy())));
   }

   /**
    * Creates a new dataset from the given stream of instances creating a new feature and label encoder from this
    * dataset and a copy this dataset's preprocessors.
    *
    * @param instances the stream of instances
    * @return the dataset
    */
   protected final Dataset<T> create(MStream<T> instances) {
      return create(instances,
                    getFeatureEncoder().createNew(),
                    getLabelEncoder().createNew(),
                    new PreprocessorList<>(preprocessors));
   }

   /**
    * Creates a new dataset from the given stream of instances using the given feature and label encoder and
    * preprocessors.
    *
    * @param instances      the stream of instances
    * @param featureEncoder the feature encoder
    * @param labelEncoder   the label encoder
    * @param preprocessors  the preprocessors
    * @return the dataset
    */
   protected abstract Dataset<T> create(MStream<T> instances, Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors);

   /**
    * Encodes the example in the dataset using the dataset's encoders.
    *
    * @return this dataset
    */
   public Dataset<T> encode() {
      if (!getFeatureEncoder().isFrozen()) {
         getFeatureEncoder().fit(this);
      }
      if (!getLabelEncoder().isFrozen()) {
         getLabelEncoder().fit(this);
      }
      log.fine("Encoded {0} Features and {1} Labels", getFeatureEncoder().size(), getLabelEncoder().size());
      return this;
   }

   /**
    * Filters the dataset using the given predicate.
    *
    * @param predicate the predicate to use for filtering the dataset
    * @return the dataset
    */
   public final Dataset<T> filter(@NonNull SerializablePredicate<T> predicate) {
      return create(stream().filter(predicate));
   }

   /**
    * Creates folds for cross-validation
    *
    * @param numberOfFolds the number of folds
    * @return the TrainTestSet made of the number of folds
    */
   public TrainTestSet<T> fold(int numberOfFolds) {
      Validation.checkArgument(numberOfFolds > 0, "Number of folds must be >= 0");
      Validation.checkArgument(size() >= numberOfFolds, "Number of folds must be <= number of examples");
      TrainTestSet<T> folds = new TrainTestSet<>();
      int foldSize = size() / numberOfFolds;
      for (int i = 0; i < numberOfFolds; i++) {
         Dataset<T> train = create(getStreamingContext().empty());
         Dataset<T> test = create(getStreamingContext().empty());

         int testStart = i * foldSize;
         int testEnd = testStart + foldSize;

         test.addAll(stream(testStart, testEnd));

         if (testStart > 0) {
            train.addAll(stream(0, testStart));
         }

         if (testEnd < size()) {
            train.addAll(stream(testEnd, size()));
         }

         folds.add(TrainTestSplit.of(train, test));
      }
      return folds;
   }

   /**
    * Gets the encoder pair used by the dataset to convert feature name into indexes
    *
    * @return the encoder pair
    */
   public EncoderPair getEncoderPair() {
      return encoders;
   }

   /**
    * Gets the feature encoder.
    *
    * @return the encoder
    */
   public Encoder getFeatureEncoder() {
      return encoders.getFeatureEncoder();
   }

   /**
    * Gets the label encoder.
    *
    * @return the encoder
    */
   public LabelEncoder getLabelEncoder() {
      return encoders.getLabelEncoder();
   }

   /**
    * Gets the preprocessors.
    *
    * @return the preprocessors
    */
   public final PreprocessorList<T> getPreprocessors() {
      return preprocessors;
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
    * Gets the dataset type.
    *
    * @return the dataset type (e.g. OffHeap, InMemory, Distributed)
    */
   public abstract DatasetType getType();

   @Override
   public Iterator<T> iterator() {
      return stream().iterator();
   }

   /**
    * Creates a leave-one-out TrainTestSet
    *
    * @return the TrainTestSet
    */
   public final TrainTestSet<T> leaveOneOut() {
      return fold(size() - 1);
   }

   /**
    * Applies the given function modifying the instances of this dataset.
    *
    * @param function The function to apply to the examples
    * @return This dataset
    */
   public abstract Dataset<T> mapSelf(SerializableFunction<? super T, T> function);

   /**
    * Creates a balanced dataset by oversampling the items
    *
    * @return the balanced dataset
    */
   public Dataset<T> oversample() {
      MCounterAccumulator<Object> accumulator = getStreamingContext().counterAccumulator();
      stream().flatMap(Example::getLabelSpace)
              .forEach(accumulator::add);
      Counter<Object> fCount = accumulator.value();
      int targetCount = (int) fCount.maximumCount();
      Dataset<T> dataset = create(getStreamingContext().empty());
      for (Object label : fCount.items()) {
         MStream<T> fStream = stream()
                                 .filter(e -> e.getLabelSpace().anyMatch(label::equals))
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
    * Preprocess the dataset with the given set of preprocessors. The preprocessing is done in place, meaning the
    * instances of this dataset will be updated.
    *
    * @param preprocessors the preprocessors to use.
    * @return the dataset
    */
   public final Dataset<T> preprocess(@NonNull PreprocessorList<T> preprocessors) {
      for (Preprocessor<T> preprocessor : preprocessors) {
         preprocessor.fit(this);
         mapSelf(preprocessor::apply);
      }
      return this;
   }

   /**
    * Reads the dataset from the given resource.
    *
    * @param resource    the resource to read from
    * @param exampleType the example type
    * @return this dataset
    * @throws IOException Something went wrong reading.
    */
   @SuppressWarnings("unchecked")
   Dataset<T> read(@NonNull Resource resource, Class<T> exampleType) throws IOException {
      try (JsonReader reader = new JsonReader(resource)) {
         reader.beginDocument();
         List<T> batch = new LinkedList<>();
         preprocessors.clear();
         //preprocessors.addAll(reader.nextProperty(PreprocessorList.class).v2);
         reader.beginArray("data");
         while (reader.peek() != JsonToken.END_ARRAY) {
            batch.add(reader.nextValue(exampleType));
            if (batch.size() > 1000) {
               addAll(batch);
               batch.clear();
            }
         }
         reader.endArray();
         addAll(batch);
         reader.endDocument();
      }
      return this;
   }

   /**
    * Samples the dataset creating a new dataset of the given sample size.
    *
    * @param withReplacement the with replacement
    * @param sampleSize      the sample size
    * @return the dataset
    */
   public Dataset<T> sample(boolean withReplacement, int sampleSize) {
      Validation.checkArgument(sampleSize > 0, "Sample size must be > 0");
      return create(stream().sample(withReplacement, sampleSize).map(e -> Cast.as(e.copy())));
   }

   /**
    * Shuffles the dataset creating a new dataset.
    *
    * @return the dataset
    */
   public final Dataset<T> shuffle() {
      return shuffle(new Random());
   }

   /**
    * Shuffles the dataset creating a new one with the given random number generator.
    *
    * @param random the random number generator
    * @return the dataset
    */
   public abstract Dataset<T> shuffle(Random random);

   /**
    * The number of examples in the dataset
    *
    * @return the number of examples in the dataset
    */
   public abstract int size();

   /**
    * Creates a new dataset containing instances from the given <code>start</code> index upto the given <code>end</code>
    * index.
    *
    * @param start the starting item index (Inclusive)
    * @param end   the ending item index (Exclusive)
    * @return the dataset
    */
   public Dataset<T> slice(int start, int end) {
      return create(stream().skip(start).limit(end - start));
   }

   /**
    * Split the dataset into a train and test split.
    *
    * @param pctTrain the percentage of the dataset to use for training
    * @return A TestTrainSet of one TestTrain item
    */
   public TrainTestSet<T> split(double pctTrain) {
      Validation.checkArgument(pctTrain > 0 && pctTrain < 1, "Percentage should be between 0 and 1");
      int split = (int) Math.floor(pctTrain * size());
      TrainTestSet<T> set = new TrainTestSet<>();
      set.add(TrainTestSplit.of(slice(0, split), slice(split, size())));
      return set;
   }

   /**
    * Creates a stream of the instances in this dataset
    *
    * @return the stream
    */
   public abstract MStream<T> stream();

   /**
    * Slices the dataset into a sub stream
    *
    * @param start the starting item index (Inclusive)
    * @param end   the ending item index (Exclusive)
    * @return the stream
    */
   protected MStream<T> stream(int start, int end) {
      return stream().skip(start).limit(end - start).cache();
   }

   /**
    * Takes the first n elements from the dataset
    *
    * @param n the number of items to take
    * @return the list of items
    */
   public List<T> take(int n) {
      return stream().take(n);
   }

   /**
    * Creates a balanced dataset by undersampling the items
    *
    * @return the balanced dataset
    */
   public Dataset<T> undersample() {
      MCounterAccumulator<Object> accumulator = getStreamingContext().counterAccumulator();
      stream().parallel()
              .flatMap(Example::getLabelSpace)
              .forEach(accumulator::add);
      Counter<Object> fCount = accumulator.value();
      int targetCount = (int) fCount.minimumCount();
      Dataset<T> dataset = create(getStreamingContext().empty());
      for (Object label : fCount.items()) {
         MStream<T> sample = stream().filter(e -> e.getLabelSpace().anyMatch(label::equals)).sample(false, targetCount);
         dataset.addAll(sample);
      }
      return dataset;
   }

   public SerializableSupplier<MStream<NDArray>> vectorStream(boolean cacheData) {
      final MStream<NDArray> mStream;
      if (cacheData) {
         mStream = asVectors().cache();
      } else {
         mStream = null;
      }

      SerializableSupplier<MStream<NDArray>> dataSupplier;
      if (mStream != null) {
         dataSupplier = () -> mStream;
      } else {
         dataSupplier = this::asVectors;
      }
      return dataSupplier;
   }

   public SerializableSupplier<MStream<NDArray>> vectorStream(boolean cacheData, double trueLabel) {
      final MStream<NDArray> mStream;
      if (cacheData) {
         mStream = asVectors(trueLabel).cache();
      } else {
         mStream = null;
      }

      SerializableSupplier<MStream<NDArray>> dataSupplier;
      if (mStream != null) {
         dataSupplier = () -> mStream;
      } else {
         dataSupplier = () -> asVectors(trueLabel);
      }
      return dataSupplier;
   }

   /**
    * Writes the dataset to the given resource in JSON format.
    *
    * @param resource the resource
    * @throws IOException Something went wrong writing the dataset
    */
   public void write(@NonNull Resource resource) throws IOException {
      try (JsonWriter writer = new JsonWriter(resource)) {
         writer.beginDocument();
         writer.property("preprocessors", preprocessors);
         writer.property("data", this);
         writer.endDocument();
      }
   }


}//END OF Dataset
