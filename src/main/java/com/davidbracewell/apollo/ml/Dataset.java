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
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.conversion.Val;
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
public interface Dataset extends Iterable<Instance>, Copyable<Dataset> {

  static DatasetBuilder builder() {
    return new DatasetBuilder();
  }

  /**
   * Add.
   *
   * @param instance the instance
   */
  void add(Instance instance);

  void addAll(MStream<Instance> stream);

  /**
   * Add all.
   *
   * @param instances the instances
   */
  void addAll(Iterable<Instance> instances);

  /**
   * Split tuple 2.
   *
   * @param pctTrain the pct train
   * @return the tuple 2
   */
  Tuple2<Dataset, Dataset> split(double pctTrain);

  /**
   * Fold list.
   *
   * @param numberOfFolds the number of folds
   * @return the list
   */
  List<Tuple2<Dataset, Dataset>> fold(int numberOfFolds);

  /**
   * Leave one out list.
   *
   * @return the list
   */
  default List<Tuple2<Dataset, Dataset>> leaveOneOut() {
    return fold(size() - 1);
  }

  /**
   * Stream m stream.
   *
   * @return the m stream
   */
  MStream<Instance> stream();

  /**
   * Gets feature encoder.
   *
   * @return the feature encoder
   */
  FeatureEncoder getFeatureEncoder();

  /**
   * Gets label encoder.
   *
   * @return the label encoder
   */
  LabelEncoder getLabelEncoder();

  /**
   * Shuffle.
   *
   * @return the dataset
   */
  Dataset shuffle();

  /**
   * Size int.
   *
   * @return the int
   */
  int size();

  @Override
  default Iterator<Instance> iterator() {
    return stream().iterator();
  }

  /**
   * To vectors stream m stream.
   *
   * @return the m stream
   */
  default MStream<Vector> toVectorsStream() {
    return stream().map(instance -> getFeatureEncoder().toVector(instance));
  }

  Dataset sample(int sampleSize);

  default void write(@NonNull Resource resource) throws IOException {
    try (JSONWriter writer = new JSONWriter(resource, true)) {
      writer.beginDocument();
      for (Instance instance : this) {
        writer.beginObject();
        writer.writeKeyValue("label", instance.getLabel());
        writer.beginObject("features");
        for (Feature feature : instance) {
          writer.writeKeyValue(feature.getName(), feature.getValue());
        }
        writer.endObject();
        writer.endObject();
      }
      writer.endDocument();
    }
  }

  default Dataset read(@NonNull Resource resource) throws IOException {
    try (JSONReader reader = new JSONReader(resource)) {
      reader.beginDocument();
      List<Instance> batch = new LinkedList<>();

      while (reader.peek() != ElementType.END_DOCUMENT) {
        reader.beginObject();
        String label = reader.nextKeyValue("label").getValue().asString();
        List<Feature> features = new LinkedList<>();
        reader.beginObject("features");
        while (reader.peek() != ElementType.END_OBJECT) {
          Tuple2<String, Val> kv = reader.nextKeyValue();
          features.add(Feature.real(kv.getKey(), kv.getValue().asDoubleValue()));
        }
        reader.endObject();
        reader.endObject();
        batch.add(Instance.create(features, label));
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

  enum Type {
    Distributed,
    InMemory,
    OffHeap
  }

}//END OF Dataset
