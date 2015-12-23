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

package com.davidbracewell.apollo.lsh;


import com.davidbracewell.apollo.linalg.KeyedVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.io.Committable;
import com.davidbracewell.tuple.Tuple2;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * The type Vector store.
 *
 * @param <KEY> the type parameter
 * @param <V>   the type parameter
 * @author David B. Bracewell
 */
public abstract class VectorStore<KEY, V extends Vector> implements Serializable, AutoCloseable, Committable {
  private static final long serialVersionUID = 1L;

  /**
   * Get v.
   *
   * @param key the key
   * @return the v
   */
  public abstract KeyedVector<KEY> get(KEY key);

  /**
   * Nearest neighbors.
   *
   * @param key the key
   * @param K   the k
   * @return the list
   */
  public final List<Tuple2<KEY, Double>> nearestNeighbors(KEY key, int K) {
    KeyedVector<KEY> vector = get(key);
    if (vector == null) {
      return Collections.emptyList();
    }
    return nearestNeighbors(vector, K);
  }

  /**
   * Nearest neighbors.
   *
   * @param query the query
   * @param K     the k
   * @return the list
   */
  public abstract List<Tuple2<KEY, Double>> nearestNeighbors(final Vector query, int K);

  /**
   * Put void.
   *
   * @param key    the key
   * @param vector the vector
   */
  public abstract void put(KEY key, V vector);

  /**
   * Similar list.
   *
   * @param query the query
   * @return the list
   */
  public abstract List<KEY> similar(final Vector query);

  public abstract boolean contains(KEY key);

  /**
   * Similar list.
   *
   * @param key the key
   * @return the list
   */
  public final List<KEY> similar(KEY key) {
    KeyedVector<KEY> vector = get(key);
    if (vector == null) {
      return Collections.emptyList();
    }
    return similar(vector);
  }

  public abstract Set<KEY> keySet();

}//END OF VectorStore
