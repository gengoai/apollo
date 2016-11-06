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

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.ScoredLabelVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.google.common.collect.MinMaxPriorityQueue;
import lombok.NonNull;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The type Lsh vector store.
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
public abstract class LSHVectorStore<KEY> implements VectorStore<KEY>, Serializable {
  protected final LSH lsh;


  /**
   * Instantiates a new Lsh vector store.
   *
   * @param lsh the lsh
   */
  public LSHVectorStore(@NonNull LSH lsh) {
    this.lsh = lsh;
  }

  /**
   * Add.
   *
   * @param vector the vector
   */
  @Override
  public final void add(@NonNull LabeledVector vector) {
    int id;
    if (containsKey(vector.getLabel())) {
      id = getID(vector.getLabel());
      remove(vector);
    } else {
      id = nextUniqueID();
    }
    registerVector(vector, id);
    lsh.add(vector, id);
  }

  @Override
  public final boolean remove(LabeledVector vector) {
    if (containsKey(vector.getLabel())) {
      int id = getID(vector.getLabel());
      lsh.remove(getVectorByID(id), id);
      removeVector(vector, id);
      return true;
    }
    return false;
  }

  protected abstract void removeVector(LabeledVector vector, int id);

  /**
   * Register vector.
   *
   * @param vector the vector
   * @param id     the id
   */
  protected abstract void registerVector(LabeledVector vector, int id);

  /**
   * Next unique id int.
   *
   * @return the int
   */
  protected abstract int nextUniqueID();

  /**
   * Gets id.
   *
   * @param key the key
   * @return the id
   */
  protected abstract int getID(KEY key);

  /**
   * Gets vector by id.
   *
   * @param id the id
   * @return the vector by id
   */
  protected abstract LabeledVector getVectorByID(int id);

  /**
   * Add.
   *
   * @param key    the key
   * @param vector the vector
   */
  public final void add(@NonNull KEY key, @NonNull Vector vector) {
    add(new LabeledVector(key, vector));
  }

  @Override
  public abstract int size();

  @Override
  public final int dimension() {
    return lsh.getDimension();
  }

  private final List<LabeledVector> query(@NonNull Vector vector) {
    OpenIntHashSet ids = lsh.query(vector);
    List<LabeledVector> vectors = new ArrayList<>();
    ids.forEachKey(id -> vectors.add(getVectorByID(id)));
    return vectors;
  }

  @Override
  public final List<ScoredLabelVector> nearest(@NonNull Vector vector, double threshold) {
    final Measure measure = lsh.getMeasure();
    return query(vector).stream()
      .map(v -> new ScoredLabelVector(v, measure.calculate(v, vector)))
      .filter(v -> lsh.getSignatureFunction().getMeasure().getOptimum().test(v.getScore(), threshold))
      .collect(Collectors.toList());
  }

  @Override
  public final List<ScoredLabelVector> nearest(@NonNull Vector vector, int K, double threshold) {
    MinMaxPriorityQueue<ScoredLabelVector> queue = MinMaxPriorityQueue
      .orderedBy(ScoredLabelVector.comparator(lsh.getOptimum()))
      .maximumSize(K)
      .create();
    queue.addAll(nearest(vector, threshold));
    List<ScoredLabelVector> list = new ArrayList<>();
    while (!queue.isEmpty()) {
      list.add(queue.remove());
    }
    return list;
  }


  @Override
  public final List<ScoredLabelVector> nearest(@NonNull Vector vector) {
    return nearest(vector, lsh.getOptimum().startingValue());
  }

  @Override
  public final List<ScoredLabelVector> nearest(@NonNull Vector vector, int K) {
    return nearest(vector, K, lsh.getOptimum().startingValue());
  }

  @Override
  public final LabeledVector get(KEY key) {
    if (containsKey(key)) {
      return getVectorByID(getID(key));
    }
    return null;
  }

}// END OF LSHVectorStore
