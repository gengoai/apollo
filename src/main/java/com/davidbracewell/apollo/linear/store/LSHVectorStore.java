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

import com.davidbracewell.apollo.hash.LSH;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.stat.measure.Measure;
import com.davidbracewell.guava.common.collect.MinMaxPriorityQueue;
import lombok.Getter;
import lombok.NonNull;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>Abstract base interface for LSH based NDArray stores.</p>
 *
 * @param <KEY> the type of key associated with vectors
 * @author David B. Bracewell
 */
public abstract class LSHVectorStore<KEY> implements VectorStore<KEY>, Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The LSH function to use
    */
   protected final LSH lsh;


   /**
    * Instantiates a new Lsh NDArray store.
    *
    * @param lsh the lsh to use
    */
   protected LSHVectorStore(@NonNull LSH lsh) {
      this.lsh = lsh;
   }

   @Override
   public final void add(@NonNull NDArray NDArray) {
      int id;
      if (NDArray.getLabel() == null) {
         return;
      }
      if (containsKey(NDArray.getLabel())) {
         id = getID(NDArray.getLabel());
         remove(NDArray);
      } else {
         id = nextUniqueID();
      }
      registerVector(NDArray, id);
      lsh.add(NDArray, id);
   }

   @Override
   public final int dimension() {
      return lsh.getDimension();
   }

   @Override
   public final NDArray get(KEY key) {
      if (containsKey(key)) {
         return getVectorByID(getID(key));
      }
      return null;
   }

   /**
    * Gets the id of a NDArray by key.
    *
    * @param key the key
    * @return the id
    */
   protected abstract int getID(KEY key);

   /**
    * Gets the NDArray by its id.
    *
    * @param id the id
    * @return the NDArray by id
    */
   protected abstract NDArray getVectorByID(int id);

   @Override
   public final List<NDArray> nearest(@NonNull NDArray query, double threshold) {
      final Measure measure = lsh.getMeasure();
      return query(query).stream()
                         .map(v -> v.copy().setWeight(measure.calculate(v, query)))
                         .filter(
                            v -> lsh.getSignatureFunction().getMeasure().getOptimum().test(v.getWeight(), threshold))
                         .collect(Collectors.toList());
   }

   @Override
   public final List<NDArray> nearest(@NonNull NDArray query, int K, double threshold) {
      MinMaxPriorityQueue<NDArray> queue = MinMaxPriorityQueue
                                              .<NDArray>orderedBy(
                                                 (v1, v2) -> lsh.getOptimum().compare(v1.getWeight(), v2.getWeight()))
                                              .maximumSize(K)
                                              .create();
      queue.addAll(nearest(query, threshold));
      List<NDArray> list = new ArrayList<>();
      while (!queue.isEmpty()) {
         list.add(queue.remove());
      }
      return list;
   }

   @Override
   public final List<NDArray> nearest(@NonNull NDArray query) {
      return nearest(query, lsh.getOptimum().startingValue());
   }

   @Override
   public final List<NDArray> nearest(@NonNull NDArray query, int K) {
      return nearest(query, K, lsh.getOptimum().startingValue());
   }

   /**
    * Gets the next unique id for assigning to vectors
    *
    * @return the int
    */
   protected abstract int nextUniqueID();

   private List<NDArray> query(@NonNull NDArray NDArray) {
      OpenIntHashSet ids = lsh.query(NDArray);
      List<NDArray> vectors = new ArrayList<>();
      ids.forEachKey(id -> vectors.add(getVectorByID(id)));
      return vectors;
   }

   /**
    * Register NDArray implementation.
    *
    * @param NDArray the NDArray
    * @param id      the id
    */
   protected abstract void registerVector(NDArray NDArray, int id);

   @Override
   public final boolean remove(NDArray NDArray) {
      if (containsKey(NDArray.getLabel())) {
         int id = getID(NDArray.getLabel());
         lsh.remove(getVectorByID(id), id);
         removeVector(NDArray, id);
         return true;
      }
      return false;
   }

   /**
    * Remove NDArray implementation.
    *
    * @param NDArray the NDArray
    * @param id      the id
    */
   protected abstract void removeVector(NDArray NDArray, int id);

   @Override
   public abstract int size();

   public static class Builder<KEY> extends VectorStoreBuilder<KEY> {
      private final LSH.Builder lshBuilder;
      @Getter
      private int bands = 5;
      @Getter
      private int buckets = 20;

      public Builder(LSH.Builder lshBuilder) {
         this.lshBuilder = lshBuilder;
      }


      @Override
      public VectorStore<KEY> build() throws IOException {
         return lshBuilder.bands(bands)
                          .buckets(buckets)
                          .dimension(getDimension())
                          .createVectorStore();
      }
   }

}// END OF LSHVectorStore
