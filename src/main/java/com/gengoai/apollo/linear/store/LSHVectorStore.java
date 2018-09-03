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

import com.gengoai.apollo.hash.LocalitySensitiveHash;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Iterators;
import com.gengoai.math.Optimum;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * <p>Abstract base interface for LSH based NDArray stores.</p>
 *
 * @param <KEY> the type of key associated with vectors
 * @author David B. Bracewell
 */
public class LSHVectorStore<KEY> implements VectorStore<KEY>, Serializable {
   private static final long serialVersionUID = 1L;
   private final LocalitySensitiveHash lsh;
   private final Map<KEY, NDArray> keyVectorMap;


   /**
    * Instantiates a new Lsh NDArray store.
    *
    * @param lsh          the lsh to use
    * @param keyVectorMap the key vector map
    */
   protected LSHVectorStore(LocalitySensitiveHash lsh, Map<KEY, NDArray> keyVectorMap) {
      this.lsh = lsh;
      this.keyVectorMap = keyVectorMap;
   }

   /**
    * Builder builder.
    *
    * @param <KEY> the type parameter
    * @return the builder
    */
   public static <KEY> Builder<KEY> builder() {
      return new Builder<>();
   }

   @Override
   public boolean containsKey(KEY key) {
      return keyVectorMap.containsKey(key);
   }

   @Override
   public final int dimension() {
      return lsh.getDimension();
   }

   @Override
   public final NDArray get(KEY key) {
      return keyVectorMap.getOrDefault(key, NDArrayFactory.SPARSE.zeros(lsh.getDimension()));
   }

   @Override
   public Measure getQueryMeasure() {
      return lsh.getMeasure();
   }

   @Override
   public Iterator<NDArray> iterator() {
      return Iterators.unmodifiableIterator(keyVectorMap.values().iterator());
   }

   @Override
   public Set<KEY> keySet() {
      return Collections.unmodifiableSet(keyVectorMap.keySet());
   }

   @Override
   public final List<NDArray> nearest(@NonNull NDArray query, double threshold) {
      final Measure measure = lsh.getMeasure();
      final Optimum optimum = measure.getOptimum();
      return lsh.query(query).stream()
                .map(v -> v.copy().setWeight(measure.calculate(v, query)))
                .filter(v -> optimum.test(v.getWeight(), threshold))
                .collect(Collectors.toList());
   }

   @Override
   public final List<NDArray> nearest(@NonNull NDArray query, int K, double threshold) {
      List<NDArray> nearest = nearest(query, threshold);
      nearest.sort((v1, v2) -> lsh.getMeasure()
                                  .getOptimum()
                                  .compare(v1.getWeight(), v2.getWeight()));
      return nearest.subList(0, Math.min(nearest.size(), K));
   }

   @Override
   public final List<NDArray> nearest(@NonNull NDArray query) {
      return nearest(query, lsh.getOptimum().startingValue());
   }

   @Override
   public final List<NDArray> nearest(@NonNull NDArray query, int K) {
      return nearest(query, K, lsh.getOptimum().startingValue());
   }

   @Override
   public int size() {
      return keyVectorMap.size();
   }

   @Override
   public VectorStoreBuilder<KEY> toBuilder() {
      return new LSHVectorStore.Builder<>(lsh.toBuilder());
   }

   /**
    * The type Builder.
    *
    * @param <KEY> the type parameter
    */
   public static class Builder<KEY> extends VectorStoreBuilder<KEY> {
      private final LocalitySensitiveHash.Builder lshBuilder;

      /**
       * Instantiates a new Builder.
       */
      public Builder() {
         this.lshBuilder = LocalitySensitiveHash.builder();
      }

      /**
       * Instantiates a new Builder.
       *
       * @param lsh the lsh
       */
      public Builder(LocalitySensitiveHash.Builder lsh) {
         this.lshBuilder = lsh;
      }

      /**
       * Bands builder.
       *
       * @param bands the bands
       * @return the builder
       */
      public Builder<KEY> bands(int bands) {
         lshBuilder.bands(bands);
         return this;
      }

      /**
       * Buckets builder.
       *
       * @param buckets the buckets
       * @return the builder
       */
      public Builder<KEY> buckets(int buckets) {
         lshBuilder.buckets(buckets);
         return this;
      }

      @Override
      public VectorStore<KEY> build() throws IOException {
         LocalitySensitiveHash lsh = lshBuilder.dimension(dimension()).inMemory();
         Map<KEY, NDArray> keyVector = new HashMap<>();
         vectors.forEach((key, vector) -> {
            lsh.add(vector);
         });
         return new LSHVectorStore<>(lsh, keyVector);
      }

      /**
       * Param builder.
       *
       * @param key   the key
       * @param value the value
       * @return the builder
       */
      public Builder<KEY> param(String key, Number value) {
         lshBuilder.param(key, value);
         return this;
      }

      /**
       * Signature builder.
       *
       * @param signature the signature
       * @return the builder
       */
      public Builder<KEY> signature(String signature) {
         lshBuilder.signature(signature);
         return this;
      }

      /**
       * Threshold builder.
       *
       * @param threshold the threshold
       * @return the builder
       */
      public Builder<KEY> threshold(double threshold) {
         lshBuilder.threshold(threshold);
         return this;
      }
   }
}// END OF LSHVectorStore
