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

import com.davidbracewell.stream.MStream;
import com.davidbracewell.tuple.Tuple2;

import java.util.List;

/**
 * The type Dataset.
 *
 * @author David B. Bracewell
 */
public interface Dataset extends Iterable<Instance> {

  /**
   * Add.
   *
   * @param instance the instance
   */
  void add(Instance instance);

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
   */
  void shuffle();

}//END OF Dataset
