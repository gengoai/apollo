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
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.guava.common.collect.MinMaxPriorityQueue;
import lombok.NonNull;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>Abstract base interface for LSH based vector stores.</p>
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
    * Instantiates a new Lsh vector store.
    *
    * @param lsh the lsh to use
    */
   protected LSHVectorStore(@NonNull LSH lsh) {
      this.lsh = lsh;
   }

   @Override
   public final void add(@NonNull Vector vector) {
      int id;
      if (vector.getLabel() == null) {
         return;
      }
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
   public final int dimension() {
      return lsh.getDimension();
   }

   @Override
   public final Vector get(KEY key) {
      if (containsKey(key)) {
         return getVectorByID(getID(key));
      }
      return null;
   }

   /**
    * Gets the id of a vector by key.
    *
    * @param key the key
    * @return the id
    */
   protected abstract int getID(KEY key);

   /**
    * Gets the vector by its id.
    *
    * @param id the id
    * @return the vector by id
    */
   protected abstract Vector getVectorByID(int id);

   @Override
   public final List<Vector> nearest(@NonNull Vector query, double threshold) {
      final Measure measure = lsh.getMeasure();
      return query(query).stream()
                         .map(v -> v.copy().setWeight(measure.calculate(v, query)))
                         .filter(
                            v -> lsh.getSignatureFunction().getMeasure().getOptimum().test(v.getWeight(), threshold))
                         .collect(Collectors.toList());
   }

   @Override
   public final List<Vector> nearest(@NonNull Vector query, int K, double threshold) {
      MinMaxPriorityQueue<Vector> queue = MinMaxPriorityQueue
                                             .<Vector>orderedBy(
                                                (v1, v2) -> lsh.getOptimum().compare(v1.getWeight(), v2.getWeight()))
                                             .maximumSize(K)
                                             .create();
      queue.addAll(nearest(query, threshold));
      List<Vector> list = new ArrayList<>();
      while (!queue.isEmpty()) {
         list.add(queue.remove());
      }
      return list;
   }

   @Override
   public final List<Vector> nearest(@NonNull Vector query) {
      return nearest(query, lsh.getOptimum().startingValue());
   }

   @Override
   public final List<Vector> nearest(@NonNull Vector query, int K) {
      return nearest(query, K, lsh.getOptimum().startingValue());
   }

   /**
    * Gets the next unique id for assigning to vectors
    *
    * @return the int
    */
   protected abstract int nextUniqueID();

   private List<Vector> query(@NonNull Vector vector) {
      OpenIntHashSet ids = lsh.query(vector);
      List<Vector> vectors = new ArrayList<>();
      ids.forEachKey(id -> vectors.add(getVectorByID(id)));
      return vectors;
   }

   /**
    * Register vector implementation.
    *
    * @param vector the vector
    * @param id     the id
    */
   protected abstract void registerVector(Vector vector, int id);

   @Override
   public final boolean remove(Vector vector) {
      if (containsKey(vector.getLabel())) {
         int id = getID(vector.getLabel());
         lsh.remove(getVectorByID(id), id);
         removeVector(vector, id);
         return true;
      }
      return false;
   }

   /**
    * Remove vector implementation.
    *
    * @param vector the vector
    * @param id     the id
    */
   protected abstract void removeVector(Vector vector, int id);

   @Override
   public abstract int size();

}// END OF LSHVectorStore
