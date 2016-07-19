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

package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.Copyable;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.RealEncoder;
import com.davidbracewell.apollo.ml.TrainTest;
import com.davidbracewell.apollo.ml.TrainTestSet;
import com.davidbracewell.apollo.ml.preprocess.Preprocessor;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.sequence.FeatureVectorSequence;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializablePredicate;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * <p>A dataset is a collection of examples which can be used for training and evaluating models. Implementations of
 * dataset may store examples in memory, off heap, or distributed using Spark.</p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public abstract class Dataset<T extends Example> implements Iterable<T>, Copyable<Dataset>, Serializable {
  private static final long serialVersionUID = 1L;

  private final EncoderPair encoders;
  private final PreprocessorList<T> preprocessors;

  /**
   * Instantiates a new Dataset.
   *
   * @param featureEncoder the feature encoder
   * @param labelEncoder   the label encoder
   * @param preprocessors  the preprocessors
   */
  protected Dataset(Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors) {
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
    return new DatasetBuilder<>(new IndexEncoder(), Instance.class);
  }


  /**
   * Creates a dataset builder with an <code>IndexEncoder</code> for the class labels as is required for classification
   * problems.
   *
   * @return the dataset builder
   */
  public static DatasetBuilder<Sequence> sequence() {
    return new DatasetBuilder<>(new IndexEncoder(), Sequence.class);
  }


  /**
   * Creates a dataset builder with a <code>RealEncoder</code> for the class labels as is required for regression
   * problems.
   *
   * @return the dataset builder
   */
  public static DatasetBuilder<Instance> regression() {
    return new DatasetBuilder<>(new RealEncoder(), Instance.class);
  }

  /**
   * Has the dataset been preprocessed?
   *
   * @return True the dataset has been preprocessed
   */
  public final boolean isPreprocessed() {
    return !preprocessors.isEmpty();
  }


  /**
   * Gets type.
   *
   * @return the type
   */
  public abstract DatasetType getType();

  /**
   * Encode dataset.
   *
   * @return the dataset
   */
  public Dataset<T> encode() {
    iterator().forEachRemaining(e -> {
    });
    return this;
  }

  @Override
  public final Iterator<T> iterator() {
    return Iterators.transform(rawIterator(), encoders::encode);
  }

  /**
   * Close.
   */
  public void close() {

  }

  /**
   * Raw un processed iterator.
   *
   * @return the iterator
   */
  protected Iterator<T> rawIterator() {
    return stream().iterator();
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
   * Preprocess the dataset with the given set of preprocessors. An <code>IllegalStateException</code> is thrown if the
   * dataset has already been processed.
   *
   * @param preprocessors the preprocessors to use.
   * @return the dataset
   */
  public Dataset<T> preprocess(@NonNull PreprocessorList<T> preprocessors) {
    Dataset<T> dataset = this;
    for (Preprocessor<T> preprocessor : preprocessors) {
      dataset = dataset.preprocess(preprocessor);
    }
    return dataset;
  }


  public Dataset<T> filter(@NonNull SerializablePredicate<T> predicate) {
    return create(stream().filter(predicate));
  }


  /**
   * Preprocess dataset.
   *
   * @param preprocessor the preprocessor
   * @return the dataset
   */
  public Dataset<T> preprocess(Preprocessor<T> preprocessor) {
    if (preprocessor == null) {
      return this;
    }
    preprocessor.fit(this);
    PreprocessorList<T> preprocessorList = new PreprocessorList<>(getPreprocessors());
    preprocessorList.add(preprocessor);
    return create(
      stream().map(preprocessor::apply),
      getFeatureEncoder().createNew(),
      getLabelEncoder().createNew(),
      preprocessorList
    );
  }


  /**
   * Creates a new dataset from the given stream of instances creating a new feature and label encoder from this
   * dataset
   * and a copy this dataset's preprocessors.
   *
   * @param instances the stream of instances
   * @return the dataset
   */
  protected final Dataset<T> create(MStream<T> instances) {
    return create(instances, getFeatureEncoder().createNew(), getLabelEncoder().createNew(), preprocessors);
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
  protected abstract Dataset<T> create(MStream<T> instances, Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors);

  /**
   * Slices the dataset int a sub stream
   *
   * @param start the start
   * @param end   the end
   * @return the m stream
   */
  public Dataset<T> slice(int start, int end) {
    return create(stream().skip(start).limit(end - start));
  }

  /**
   * Slices the dataset int a sub stream
   *
   * @param start the start
   * @param end   the end
   * @return the m stream
   */
  protected MStream<T> streamSlice(int start, int end) {
    return stream().skip(start).limit(end - start);
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
   * Add all.
   *
   * @param instances the instances
   */
  @SafeVarargs
  protected final void addAll(@NonNull T... instances) {
    addAll(Arrays.asList(instances));
  }


  /**
   * Split the dataset into a train and test split.
   *
   * @param pctTrain the percentage of the dataset to use for training
   * @return A TestTrainSet of one TestTrain item
   */
  public TrainTestSet<T> split(double pctTrain) {
    Preconditions.checkArgument(pctTrain > 0 && pctTrain < 1, "Percentage should be between 0 and 1");
    int split = (int) Math.floor(pctTrain * size());
    TrainTestSet<T> set = new TrainTestSet<>();
    set.add(TrainTest.of(slice(0, split), slice(split, size())));
    set.trimToSize();
    return set;
  }

  @Override
  public Dataset<T> copy() {
    return create(stream().map(e -> Cast.as(e.copy())));
  }

  /**
   * Creates folds for cross-validation
   *
   * @param numberOfFolds the number of folds
   * @return the TrainTestSet made of the number of folds
   */
  public TrainTestSet<T> fold(int numberOfFolds) {
    Preconditions.checkArgument(numberOfFolds > 0, "Number of folds must be >= 0");
    Preconditions.checkArgument(size() >= numberOfFolds, "Number of folds must be <= number of examples");
    TrainTestSet<T> folds = new TrainTestSet<>();

    int foldSize = size() / numberOfFolds;
    for (int i = 0; i < numberOfFolds; i++) {
      MStream<T> train;
      MStream<T> test;
      if (i == 0) {
        test = streamSlice(0, foldSize);
        train = streamSlice(foldSize, size());
      } else if (i == numberOfFolds - 1) {
        test = streamSlice(size() - foldSize, size());
        train = streamSlice(0, size() - foldSize);
      } else {
        train = streamSlice(0, foldSize * i).union(streamSlice(foldSize * i + foldSize, size()));
        test = streamSlice(foldSize * i, foldSize * i + foldSize);
      }
      folds.add(TrainTest.of(create(train), create(test)));
    }

    folds.trimToSize();
    return folds;
  }

  /**
   * Samples the dataset creating a new dataset of the given sample size.
   *
   * @param sampleSize the sample size
   * @return the dataset
   */
  public Dataset<T> sample(int sampleSize) {
    Preconditions.checkArgument(sampleSize > 0, "Sample size must be > 0");
    return create(stream().sample(sampleSize).map(e -> Cast.as(e.copy())));
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
  public Encoder getLabelEncoder() {
    return encoders.getLabelEncoder();
  }

  /**
   * Gets encoder pair.
   *
   * @return the encoder pair
   */
  public EncoderPair getEncoderPair() {
    return encoders;
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
   * Creates a stream of the instances
   *
   * @return the stream
   */
  public abstract MStream<T> stream();

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
   * Writes the dataset to the given resource in JSON format.
   *
   * @param resource the resource
   * @throws IOException Something went wrong writing the dataset
   */
  public void write(@NonNull Resource resource) throws IOException {
    try (JSONWriter writer = new JSONWriter(resource)) {
      writer.beginDocument();
      writer.beginArray("preprocessors");
      for (Preprocessor<T> preprocessor : preprocessors) {
        writer.writeValue(preprocessor.describe());
      }
      writer.endArray();
      writer.beginArray("data");
      for (T instance : this) {
        writer.writeValue(instance);
      }
      writer.endArray();
      writer.endDocument();
    }
  }


  /**
   * Reads the dataset from the given resource.
   *
   * @param resource    the resource to read from
   * @param exampleType the example type
   * @return this dataset
   * @throws IOException Something went wrong reading.
   */
  Dataset<T> read(@NonNull Resource resource, Class<T> exampleType) throws IOException {
    try (JSONReader reader = new JSONReader(resource)) {
      reader.beginDocument();
      List<T> batch = new LinkedList<>();
      reader.beginArray("preprocessors");
      while (reader.peek() != ElementType.END_ARRAY) {
        reader.nextValue();
      }
      reader.endArray();
      reader.beginArray("data");
      while (reader.peek() != ElementType.END_ARRAY) {
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
   * As feature vectors list.
   *
   * @return the list
   */
  public List<FeatureVector> asFeatureVectors() {
    encode();
    List<FeatureVector> list = stream()
      .flatMap(e -> e.asInstances().stream().map(ii -> ii.toVector(encoders)).collect(Collectors.toList()))
      .collect();
    close();
    return list;
  }

  /**
   * As feature vector sequences list.
   *
   * @return the list
   */
  public List<FeatureVectorSequence> asFeatureVectorSequences() {
    encode();
    List<FeatureVectorSequence> list = stream()
      .map(e -> {
        FeatureVectorSequence fvs = new FeatureVectorSequence();
        for (Instance ii : e.asInstances()) {
          fvs.add(ii.toVector(encoders));
        }
        return fvs;
      })
      .collect();
    close();
    return list;
  }

  /**
   * Take list.
   *
   * @param n the n
   * @return the list
   */
  public List<T> take(int n) {
    return stream().take(n);
  }

}//END OF Dataset
