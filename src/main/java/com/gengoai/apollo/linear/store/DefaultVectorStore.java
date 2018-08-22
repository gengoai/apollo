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

package com.gengoai.apollo.linear.store;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Iterators;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

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
      Validation.checkArgument(dimension > 0, "Dimension must be > 0");
      this.dimension = dimension;
      this.queryMeasure = queryMeasure;
   }

   public static <KEY> VectorStoreBuilder<KEY> builder(int dimension) {
      return new Builder<KEY>().dimension(dimension);
   }

   public static <KEY> VectorStoreBuilder<KEY> builder() {
      return new Builder<>();
   }

   @Override
   public boolean containsKey(@NonNull KEY key) {
      return vectorMap.containsKey(key);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(@NonNull KEY key) {
      return vectorMap.getOrDefault(key, NDArrayFactory.SPARSE.zeros(dimension));
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
   public Measure getQueryMeasure() {
      return queryMeasure;
   }

   @Override
   public int size() {
      return vectorMap.size();
   }

   @Override
   public VectorStoreBuilder<KEY> toBuilder() {
      return builder(dimension);
   }

   public static class Builder<KEY> extends VectorStoreBuilder<KEY> {
      @Override
      public VectorStore<KEY> build() throws IOException {
         DefaultVectorStore<KEY> vs = new DefaultVectorStore<>(dimension(), measure());
         vs.vectorMap.putAll(vectors);
         return vs;
      }
   }// END OF Builder

}//END OF DefaultVectorStore
