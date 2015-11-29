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

package com.davidbracewell.apollo.learning;

import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * The type Dataset.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class Dataset<T> implements Serializable, Iterable<Instance> {
  private static final long serialVersionUID = 1L;
  private final List<Instance> instances = new ArrayList<>();
  private final Featurizer<T> featurizer;

  /**
   * Instantiates a new Dataset.
   *
   * @param featurizer the featurizer
   */
  public Dataset(@NonNull Featurizer<T> featurizer) {
    this.featurizer = featurizer;
  }

  @Override
  public Iterator<Instance> iterator() {
    return instances.iterator();
  }

  /**
   * Stream stream.
   *
   * @return the stream
   */
  public Stream<Instance> stream() {
    return instances.stream();
  }

  /**
   * Parallel stream stream.
   *
   * @return the stream
   */
  public Stream<Instance> parallelStream() {
    return instances.parallelStream();
  }

  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return instances.size();
  }

  /**
   * Add.
   *
   * @param object the object
   */
  public void add(@NonNull T object) {
    instances.add(Instance.create(featurizer.apply(object)));
  }

  /**
   * Add.
   *
   * @param object the object
   * @param label  the label
   */
  public void add(@NonNull T object, Object label) {
    instances.add(Instance.create(featurizer.apply(object), label));
  }

  /**
   * Add.
   *
   * @param object        the object
   * @param labelFunction the label function
   */
  public void add(@NonNull T object, @NonNull Function<? super T, ?> labelFunction) {
    instances.add(Instance.create(featurizer.apply(object), labelFunction.apply(object)));
  }


}//END OF Dataset
