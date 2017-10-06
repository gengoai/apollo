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

package com.davidbracewell.apollo.linear.store;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.stat.measure.Measure;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.collect.Iterators;
import lombok.NonNull;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * The type Default vector store.
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
public class DefaultVectorStore<KEY> implements VectorStore<KEY>, Serializable {
   private static final long serialVersionUID = 1L;
   private final Map<KEY, NDArray> vectorMap = new HashMap<>();
   private int dimension;
   private Measure queryMeasure;

   /**
    * Instantiates a new Default vector store.
    *
    * @param dimension    the dimension
    * @param queryMeasure the query measure
    */
   public DefaultVectorStore(int dimension, @NonNull Measure queryMeasure) {
      Preconditions.checkArgument(dimension > 0, "Dimension must be > 0");
      this.dimension = dimension;
      this.queryMeasure = queryMeasure;
   }

   @Override
   public void add(@NonNull NDArray vector) {
      Preconditions.checkArgument(vector.length() == dimension,
                                  "Dimension mismatch, vector store can only store vectors with k of " + dimension);
      vectorMap.put(vector.getLabel(), vector);
   }

   @Override
   public boolean containsKey(@NonNull KEY key) {
      return vectorMap.containsKey(key);
   }

   @Override
   public VectorStore<KEY> createNew() {
      return new DefaultVectorStore<>(dimension, queryMeasure);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(@NonNull KEY key) {
      return vectorMap.get(key);
   }

   @Override
   public Iterator<NDArray> iterator() {
      return Iterators.unmodifiableIterator(vectorMap.values().iterator());
   }

   @Override
   public Set<KEY> keys() {
      return Collections.unmodifiableSet(vectorMap.keySet());
   }

   @Override
   public List<NDArray> nearest(NDArray query, double threshold) {
      Preconditions.checkArgument(query.length() == dimension,
                                  "Dimension mismatch, vector store can only store vectors with k of " + dimension);
      return vectorMap.values().parallelStream()
                      .map(v -> v.copy().setWeight(queryMeasure.calculate(v, query)))
                      .filter(s -> queryMeasure.getOptimum().test(s.getWeight(), threshold))
                      .collect(Collectors.toList());
   }

   @Override
   public List<NDArray> nearest(@NonNull NDArray query, int K, double threshold) {
      Preconditions.checkArgument(query.length() == dimension,
                                  "Dimension mismatch, vector store can only store vectors with k of " + dimension);
      List<NDArray> vectors = vectorMap.values().parallelStream()
                                       .map(v -> v.copy().setWeight(queryMeasure.calculate(v, query)))
                                       .filter(s -> queryMeasure.getOptimum().test(s.getWeight(), threshold))
                                       .sorted((s1, s2) -> queryMeasure.getOptimum()
                                                                       .compare(s1.getWeight(), s2.getWeight()))
                                       .collect(Collectors.toList());
      return vectors.subList(0, Math.min(K, vectors.size()));
   }

   @Override
   public List<NDArray> nearest(@NonNull NDArray query) {
      Preconditions.checkArgument(query.length() == dimension,
                                  "Dimension mismatch, vector store can only store vectors with k of " + dimension);
      return vectorMap.values().parallelStream()
                      .map(v -> v.copy().setWeight(queryMeasure.calculate(v, query)))
                      .collect(Collectors.toList());
   }

   @Override
   public List<NDArray> nearest(@NonNull NDArray query, int K) {
      Preconditions.checkArgument(query.length() == dimension,
                                  "Dimension mismatch, vector store can only store vectors with k of " + dimension);
      List<NDArray> vectors = vectorMap.values().parallelStream()
                                       .map(v -> v.copy().setWeight(queryMeasure.calculate(v, query)))
                                       .sorted((s1, s2) -> queryMeasure.getOptimum()
                                                                       .compare(s1.getWeight(), s2.getWeight()))
                                       .collect(Collectors.toList());
      return vectors.subList(0, Math.min(K, vectors.size()));
   }

   @Override
   public boolean remove(@NonNull NDArray vector) {
      return vector.getLabel() != null && vectorMap.remove(vector.getLabel()) != null;
   }

   @Override
   public int size() {
      return vectorMap.size();
   }

}//END OF DefaultVectorStore
