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

package com.davidbracewell.apollo.linalg.store;

import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.ScoredLabelVector;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

import java.util.List;
import java.util.Set;

/**
 * The interface Vector store.
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
public interface VectorStore<KEY> extends Iterable<LabeledVector> {

  /**
   * Size int.
   *
   * @return the int
   */
  int size();

  /**
   * Gets dimension.
   *
   * @return the dimension
   */
  int dimension();

  /**
   * Add.
   *
   * @param vector the vector
   */
  void add(@NonNull LabeledVector vector);


  /**
   * Nearest list.
   *
   * @param vector    the vector
   * @param threshold the threshold
   * @return the list
   */
  List<ScoredLabelVector> nearest(Vector vector, double threshold);


  /**
   * Nearest list.
   *
   * @param vector    the vector
   * @param K         the k
   * @param threshold the threshold
   * @return the list
   */
  List<ScoredLabelVector> nearest(@NonNull Vector vector, int K, double threshold);

  /**
   * Nearest list.
   *
   * @param vector the vector
   * @return the list
   */
  List<ScoredLabelVector> nearest(Vector vector);


  /**
   * Nearest list.
   *
   * @param vector the vector
   * @param K      the k
   * @return the list
   */
  List<ScoredLabelVector> nearest(@NonNull Vector vector, int K);


  /**
   * Key set set.
   *
   * @return the set
   */
  Set<KEY> keySet();

  /**
   * Get labeled vector.
   *
   * @param key the key
   * @return the labeled vector
   */
  LabeledVector get(KEY key);

  /**
   * Contains key boolean.
   *
   * @param key the key
   * @return the boolean
   */
  boolean containsKey(KEY key);

  /**
   * Remove boolean.
   *
   * @param vector the vector
   * @return the boolean
   */
  boolean remove(LabeledVector vector);


}// END OF VectorStore
