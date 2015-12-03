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
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * The type Dataset.
 *
 * @author David B. Bracewell
 */
public interface Dataset<T extends Example> extends Iterable<T>, Copyable<Dataset> {

  /**
   * Builder dataset builder.
   *
   * @return the dataset builder
   */
  static <T extends Example> DatasetBuilder<T> builder() {
    return new DatasetBuilder<>();
  }

  /**
   * Add.
   *
   * @param instance the instance
   */
  void add(T instance);

  /**
   * Add all.
   *
   * @param stream the stream
   */
  void addAll(MStream<T> stream);

  /**
   * Add all.
   *
   * @param instances the instances
   */
  void addAll(Iterable<T> instances);

  /**
   * Split tuple 2.
   *
   * @param pctTrain the pct train
   * @return the tuple 2
   */
  Tuple2<Dataset<T>, Dataset<T>> split(double pctTrain);

  /**
   * Fold list.
   *
   * @param numberOfFolds the number of folds
   * @return the list
   */
  List<Tuple2<Dataset<T>, Dataset<T>>> fold(int numberOfFolds);

  /**
   * Leave one out list.
   *
   * @return the list
   */
  default List<Tuple2<Dataset<T>, Dataset<T>>> leaveOneOut() {
    return fold(size() - 1);
  }

  /**
   * Stream m stream.
   *
   * @return the m stream
   */
  MStream<T> stream();

  /**
   * Gets feature encoder.
   *
   * @return the feature encoder
   */
  Encoder featureEncoder();

  /**
   * Gets label encoder.
   *
   * @return the label encoder
   */
  Encoder labelEncoder();

  /**
   * Shuffle.
   *
   * @return the dataset
   */
  Dataset<T> shuffle();

  /**
   * Size int.
   *
   * @return the int
   */
  int size();

  @Override
  default Iterator<T> iterator() {
    return stream().iterator();
  }

  /**
   * Sample dataset.
   *
   * @param sampleSize the sample size
   * @return the dataset
   */
  Dataset<T> sample(int sampleSize);

  /**
   * Write.
   *
   * @param resource the resource
   * @throws IOException the io exception
   */
  default void write(@NonNull Resource resource) throws IOException {
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
  default Dataset<T> read(@NonNull Resource resource) throws IOException {
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
