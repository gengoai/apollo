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

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.io.Committable;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.Closeable;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Implementation of Locality Sensitive Hashing
 *
 * @author David B. Bracewell
 */
public abstract class LSH<V extends Vector> implements Serializable, Committable, Closeable {

  private static final long serialVersionUID = 1L;
  private final LSHTable[] tables;
  private final HashFamily family;
  private final double threshold;

  /**
   * Instantiates a new LSH.
   *
   * @param hashFamily the hash family
   */
  public LSH(@NonNull HashFamily hashFamily, @NonNull LSHTable[] tables) {
    this.family = hashFamily;
    this.tables = tables;
    threshold = Math.pow(1.0 / tables.length, 1.0 / tables[0].getHashGroup().getNumberOfHashes());
  }

  /**
   * Add void.
   *
   * @param v the v
   */
  public void add(V v) {
    if (v != null) {
      int index = addVector(v);
      for (LSHTable table : tables) {
        table.add(v, index);
      }
    }
  }

  public final void addAll(Collection<? extends V> vectors) {
    if (vectors != null) {
      for (V v : vectors) {
        add(v);
      }
    }
  }

  protected abstract int addVector(V vector);

  protected LSHTable[] getTables() {
    return tables;
  }

  protected abstract V getVector(int index);

  /**
   * Query list.
   *
   * @param query      the query
   * @param maxResults the max results
   * @return the list
   */
  public List<Tuple2<V, Double>> nearestNeighbors(@NonNull Vector query, int maxResults) {
    DistanceMeasure dm = family.getDistanceMeasure();
    Counter<V> scores = Counters.newHashMapCounter();
    for (V vector : similar(query)) {
      scores.set(vector, dm.calculate(query, vector));
    }

    List<Tuple2<V, Double>> results = new ArrayList<>();
    for (V vector : scores.itemsByCount(true)) {
      results.add(Tuple2.of(vector, scores.get(vector)));
      if (results.size() >= maxResults) {
        break;
      }
    }

    return results;
  }

  /**
   * Query list.
   *
   * @param query the query
   * @return the list
   */
  public List<V> similar(@NonNull final Vector query) {
    Counter<Integer> candidates = Counters.newHashMapCounter();
    for (LSHTable table : tables) {
      candidates.incrementAll(table.get(query));
    }
    candidates = candidates.filterByValue(d -> d >= threshold);
    return candidates.items().stream().map(this::getVector).collect(Collectors.toList());
  }


}//END OF LSHIndex
