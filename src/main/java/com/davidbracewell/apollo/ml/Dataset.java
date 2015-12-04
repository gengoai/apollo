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

/**
 * The type Dataset.
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
   * Builder dataset builder.
   *
   * @param <T> the type parameter
   * @return the dataset builder
   */
  public static <T extends Example> DatasetBuilder<T> builder() {
    return new DatasetBuilder<>();
  }

  public static DatasetBuilder<Instance> classification() {
    return new DatasetBuilder<>();
  }

  public static DatasetBuilder<Instance> regression() {
    return new DatasetBuilder<Instance>().labelEncoder(new RealEncoder());
  }


  /**
   * Is preprocessed boolean.
   *
   * @return the boolean
   */
  public final boolean isPreprocessed() {
    return !preprocessors.isEmpty();
  }

  /**
   * Preprocess.
   *
   * @param preprocessors the preprocessors
   */
  public void preprocess(@NonNull PreprocessorList<T> preprocessors) {
    Preconditions.checkState(!isPreprocessed(), "Dataset has already been preprocessed");
    this.preprocessors = preprocessors;
    if (!this.preprocessors.isFinished()) {
      this.preprocessors.visit(this.rawIterator());
      this.preprocessors.finish();
    }
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
   * Raw iterator iterator.
   *
   * @return the iterator
   */
  protected Iterator<T> rawIterator() {
    return stream().iterator();
  }

  /**
   * Create dataset.
   *
   * @param instances the instances
   * @return the dataset
   */
  protected final Dataset<T> create(MStream<T> instances) {
    return create(instances, featureEncoder().createNew(), labelEncoder().createNew(), preprocessors);
  }

  /**
   * Create dataset.
   *
   * @param instances      the instances
   * @param featureEncoder the feature encoder
   * @param labelEncoder   the label encoder
   * @param preprocessors  the preprocessors
   * @return the dataset
   */
  protected abstract Dataset<T> create(MStream<T> instances, Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors);

  /**
   * Slice m stream.
   *
   * @param start the start
   * @param end   the end
   * @return the m stream
   */
  protected MStream<T> slice(int start, int end) {
    return stream().skip(start).limit(end - start);
  }


  /**
   * Gets preprocessors.
   *
   * @return the preprocessors
   */
  public final PreprocessorList<T> getPreprocessors() {
    return preprocessors;
  }

  /**
   * Add all.
   *
   * @param stream the stream
   */
  protected abstract void addAll(MStream<T> stream);

  /**
   * Add all.
   *
   * @param instances the instances
   */
  protected void addAll(@NonNull Collection<T> instances) {
    addAll(Streams.of(instances, false));
  }


  /**
   * Split tuple 2.
   *
   * @param pctTrain the pct train
   * @return the tuple 2
   */
  public TrainTestSet<T> split(double pctTrain) {
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
   * Fold list.
   *
   * @param numberOfFolds the number of folds
   * @return the list
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
   * Sample dataset.
   *
   * @param sampleSize the sample size
   * @return the dataset
   */
  public Dataset<T> sample(int sampleSize) {
    return create(stream().sample(sampleSize));
  }

  /**
   * Feature encoder encoder.
   *
   * @return the encoder
   */
  public Encoder featureEncoder() {
    return featureEncoder;
  }

  /**
   * Label encoder encoder.
   *
   * @return the encoder
   */
  public Encoder labelEncoder() {
    return labelEncoder;
  }


  /**
   * Leave one out list.
   *
   * @return the list
   */
  public final TrainTestSet<T> leaveOneOut() {
    return fold(size() - 1);
  }

  /**
   * Stream m stream.
   *
   * @return the m stream
   */
  public abstract MStream<T> stream();

  /**
   * Shuffle.
   *
   * @return the dataset
   */
  public final Dataset<T> shuffle() {
    return shuffle(new Random());
  }

  public abstract Dataset<T> shuffle(Random random);

  /**
   * Size int.
   *
   * @return the int
   */
  public abstract int size();

  /**
   * Write.
   *
   * @param resource the resource
   * @throws IOException the io exception
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
   * Read dataset.
   *
   * @param resource the resource
   * @return the dataset
   * @throws IOException the io exception
   */
  public Dataset<T> read(@NonNull Resource resource) throws IOException {
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
   * The enum Type.
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
