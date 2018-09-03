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

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Iterators;

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
   private final int dimension;
   private final Measure queryMeasure;

   private DefaultVectorStore(int dimension, Measure queryMeasure) {
      this.dimension = dimension;
      this.queryMeasure = queryMeasure;
   }

   /**
    * Builder vector store builder.
    *
    * @param <KEY>     the type parameter
    * @param dimension the dimension
    * @return the vector store builder
    */
   public static <KEY> VectorStoreBuilder<KEY> builder(int dimension) {
      return new Builder<KEY>().dimension(dimension);
   }

   /**
    * Builder vector store builder.
    *
    * @param <KEY> the type parameter
    * @return the vector store builder
    */
   public static <KEY> VectorStoreBuilder<KEY> builder() {
      return new Builder<>();
   }

   @Override
   public boolean containsKey(KEY key) {
      return vectorMap.containsKey(key);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(KEY key) {
      return vectorMap.getOrDefault(key, NDArrayFactory.SPARSE.zeros(dimension));
   }

   @Override
   public Iterator<NDArray> iterator() {
      return Iterators.unmodifiableIterator(vectorMap.values().iterator());
   }

   @Override
   public Set<KEY> keySet() {
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
      return DefaultVectorStore.<KEY>builder(dimension).measure(getQueryMeasure());
   }

   /**
    * The type Builder.
    *
    * @param <KEY> the type parameter
    */
   public static class Builder<KEY> extends VectorStoreBuilder<KEY> {
      @Override
      public VectorStore<KEY> build() {
         DefaultVectorStore<KEY> vs = new DefaultVectorStore<>(dimension(), measure());
         vs.vectorMap.putAll(vectors);
         return vs;
      }
   }// END OF Builder

}//END OF DefaultVectorStore
