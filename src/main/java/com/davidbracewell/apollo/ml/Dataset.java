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

package com.davidbracewell.apollo.ml;

import com.davidbracewell.Copyable;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
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

  private final Encoder featureEncoder;
  private final Encoder labelEncoder;
  private volatile PreprocessorList<T> preprocessors;

  /**
   * Instantiates a new Dataset.
   *
   * @param featureEncoder the feature encoder
   * @param labelEncoder   the label encoder
   * @param preprocessors  the preprocessors
   */
  protected Dataset(Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors) {
    this.featureEncoder = featureEncoder;
    this.labelEncoder = labelEncoder;
    this.preprocessors = preprocessors == null ? PreprocessorList.empty() : preprocessors;
  }

  /**
   * Creates a generic dataset builder which defaults to an <code>IndexEncoder</code> for the labels and features.
   *
   * @param <T> the type of example the dataset contains.
   * @return the dataset builder
   */
  public static <T extends Example> DatasetBuilder<T> builder() {
    return new DatasetBuilder<>();
  }

  /**
   * Creates a dataset builder with an <code>IndexEncoder</code> for the class labels as is required for classification
   * problems.
   *
   * @return the dataset builder
   */
  public static DatasetBuilder<Instance> classification() {
    return new DatasetBuilder<>();
  }

  /**
   * Creates a dataset builder with a <code>RealEncoder</code> for the class labels as is required for regression
   * problems.
   *
   * @return the dataset builder
   */
  public static DatasetBuilder<Instance> regression() {
    return new DatasetBuilder<Instance>().labelEncoder(new RealEncoder());
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
   * Preprocess the dataset with the given set of preprocessors. An <code>IllegalStateException</code> is thrown if the
   * dataset has already been processed.
   *
   * @param preprocessors the preprocessors to use.
   */
  public void preprocess(@NonNull PreprocessorList<T> preprocessors) {
    Preconditions.checkState(!isPreprocessed(), "Dataset has already been preprocessed");
    this.preprocessors = preprocessors;
    if (!this.preprocessors.isFinished()) {
      this.preprocessors.visit(this.rawIterator());
      this.preprocessors.finish();
    }
  }

  public void encode() {
    forEach(e -> {
      labelEncoder.encode(e.getLabelSpace());
      featureEncoder.encode(e.getFeatureSpace());
    });
  }

  @Override
  public final Iterator<T> iterator() {
    return Iterators.transform(rawIterator(), e -> {
      e = preprocessors.apply(e);
      labelEncoder.encode(e.getLabelSpace());
      featureEncoder.encode(e.getFeatureSpace());
      return e;
    });
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
   * Creates a new dataset from the given stream of instances creating a new feature and label encoder from this dataset
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
  protected MStream<T> slice(int start, int end) {
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
    addAll(Streams.of(instances, false));
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
    set.add(TrainTest.of(
      create(slice(0, split)),
      create(slice(split, size()))
    ));
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
    TrainTestSet<T> folds = new TrainTestSet<>();

    int foldSize = size() / numberOfFolds;
    for (int i = 0; i < numberOfFolds; i++) {
      MStream<T> train;
      MStream<T> test;
      if (i == 0) {
        test = slice(0, foldSize);
        train = slice(foldSize, size());
      } else if (i == numberOfFolds - 1) {
        test = slice(size() - foldSize, size());
        train = slice(0, size() - foldSize);
      } else {
        train = slice(0, foldSize * i).union(slice(foldSize * i + foldSize, size()));
        test = slice(foldSize * i, foldSize * i + foldSize);
      }

      folds.add(
        TrainTest.of(create(train), create(test))
      );
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
    return featureEncoder;
  }

  /**
   * Gets the label encoder.
   *
   * @return the encoder
   */
  public Encoder getLabelEncoder() {
    return labelEncoder;
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
    try (JSONWriter writer = new JSONWriter(resource, true)) {
      writer.beginDocument();
      for (T instance : this) {
        instance.write(writer);
      }
      writer.endDocument();
    }
  }


  /**
   * Reads the dataset from the given resource.
   *
   * @param resource the resource to read from
   * @return this dataset
   * @throws IOException Something went wrong reading.
   */
  Dataset<T> read(@NonNull Resource resource) throws IOException {
    try (JSONReader reader = new JSONReader(resource)) {
      reader.beginDocument();
      List<T> batch = new LinkedList<>();
      while (reader.peek() != ElementType.END_DOCUMENT) {
        batch.add(Cast.as(Example.read(reader)));
        if (batch.size() > 1000) {
          addAll(batch);
          batch.clear();
        }
      }
      addAll(batch);
      reader.endDocument();
    }
    return this;
  }

  /**
   * The dataset type.
   */
  enum Type {
    /**
     * Distributed type.
     */
    Distributed,
    /**
     * In memory type.
     */
    InMemory,
    /**
     * Off heap type.
     */
    OffHeap
  }

}//END OF Dataset
