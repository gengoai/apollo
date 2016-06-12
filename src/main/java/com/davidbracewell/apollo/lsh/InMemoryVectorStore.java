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

import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.HashMapIndex;
import com.davidbracewell.collection.Index;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.base.Preconditions;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class InMemoryVectorStore<KEY, V extends Vector> extends VectorStore<KEY, V> {
  private static final long serialVersionUID = 1L;
  private final InMemoryLSH<LabeledVector> lsh;
  private final Index<KEY> index;

  public InMemoryVectorStore(HashFamily hashFamily, int numberOfHashes, int numberOfBuckets) {
    Preconditions.checkArgument(numberOfHashes > 0, "Number of hashes must be > 0");
    Preconditions.checkArgument(numberOfBuckets > 0, "Number of buckets must be > 0");
    this.lsh = new InMemoryLSH<>(Preconditions.checkNotNull(hashFamily), numberOfHashes, numberOfBuckets);
    this.index = new HashMapIndex<>();
  }

  public InMemoryVectorStore(HashFamily hashFamily, int numberOfBuckets, double desiredSimilarity) {
    double y1 = 1d - Math.pow(1d - 0.9996, 1d / numberOfBuckets);
    int numberOfHashes = (int) Math.floor(Math.log10(y1) / Math.log10(desiredSimilarity));
    this.lsh = new InMemoryLSH<>(Preconditions.checkNotNull(hashFamily), numberOfHashes, numberOfBuckets);
    this.index = new HashMapIndex<>();
  }

  @Override
  public void close() throws Exception {

  }

  @Override
  public void commit() {

  }

  @Override
  public boolean contains(KEY key) {
    return index.contains(key);
  }

  @Override
  public LabeledVector get(KEY key) {
    if (key == null || !this.index.contains(key)) {
      return null;
    }
    return lsh.getVector(this.index.indexOf(key));
  }

  @Override
  public Set<KEY> keySet() {
    return new HashSet<>(index.asList());
  }

  @Override
  public List<Tuple2<KEY, Double>> nearestNeighbors(Vector query, int K) {
    if (query != null && K > 0) {
      lsh.nearestNeighbors(query, K);
    }
    return Collections.emptyList();
  }

  @Override
  public void put(KEY key, V vector) {
    if (key != null && vector != null) {
      lsh.add(new LabeledVector(key, vector));
      this.index.add(key);
    }
  }

  @Override
  public List<KEY> similar(Vector query) {
    if (query == null) {
      return Collections.emptyList();
    }
    return lsh.similar(query).stream().map(LabeledVector::getLabel).map(Cast::<KEY>as).collect(Collectors.toList());
  }
}//END OF InMemoryVectorStore
