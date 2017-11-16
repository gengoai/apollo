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

package com.davidbracewell.apollo.hash;

import com.davidbracewell.apollo.linear.store.InMemoryLSHVectorStore;
import com.davidbracewell.apollo.linear.store.VectorStore;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.function.BiFunction;

/**
 * The type Lsh.
 *
 * @author David B. Bracewell
 */
public class InMemoryLSH extends LSH {
   private static final long serialVersionUID = 1L;
   private OpenIntHashSet[][] vectorStore;


   /**
    * Instantiates a new Lsh.
    *
    * @param bands             the bands
    * @param buckets           the buckets
    * @param signatureFunction the signature function
    */
   public InMemoryLSH(int bands, int buckets, @NonNull SignatureFunction signatureFunction) {
      super(bands, buckets, signatureFunction);
      initVectorStore();
   }

   /**
    * Instantiates a new Lsh.
    *
    * @param bands             the bands
    * @param buckets           the buckets
    * @param dimension         the dimension
    * @param threshold         the threshold
    * @param signatureSupplier the signature supplier
    */
   public InMemoryLSH(int bands, int buckets, int dimension, double threshold, @NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
      super(bands, buckets, dimension, threshold, signatureSupplier);
      initVectorStore();
   }

   /**
    * Instantiates a new Lsh.
    *
    * @param bands             the bands
    * @param buckets           the buckets
    * @param dimension         the dimension
    * @param signatureSupplier the signature supplier
    */
   public InMemoryLSH(int bands, int buckets, int dimension, @NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
      this(bands, buckets, dimension, 0.5, signatureSupplier);
   }

   /**
    * Builder builder.
    *
    * @return the builder
    */
   public static Builder builder() {
      return new Builder();
   }

   private void initVectorStore() {
      this.vectorStore = new OpenIntHashSet[getBands()][getBuckets()];
      for (int b = 0; b < getBands(); b++) {
         for (int u = 0; u < getBuckets(); u++) {
            this.vectorStore[b][u] = new OpenIntHashSet();
         }
      }
   }

   @Override
   public void clear() {
      initVectorStore();
   }

   @Override
   protected OpenIntHashSet get(int band, int bucket) {
      return vectorStore[band][bucket];
   }

   @Override
   protected void addToTable(int band, int bucket, int vid) {
      vectorStore[band][bucket].add(vid);
   }

   /**
    * The type Builder.
    */
   public static class Builder extends LSH.Builder {

      @Override
      public InMemoryLSH create() {
         if (signatureFunction != null) {
            return new InMemoryLSH(bands, buckets, signatureFunction);
         }
         Preconditions.checkArgument(dimension > 0, "Dimension not set.");
         Preconditions.checkNotNull(signatureSupplier, "A signature suppler was not set");
         return new InMemoryLSH(bands, buckets, dimension, threshold, signatureSupplier);
      }

      @Override
      public <KEY> VectorStore<KEY> createVectorStore() {
         return new InMemoryLSHVectorStore<>(create());
      }

   }

}// END OF LSH
