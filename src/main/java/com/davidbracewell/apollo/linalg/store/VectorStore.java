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

import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.ScoredLabelVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.io.Commitable;
import lombok.NonNull;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.Set;

/**
 * <p>A vector store provides access and lookup of vectors by labels and to find vectors in the store closest to query
 * vectors. </p>
 *
 * @param <KEY> the type of key associated with vectors
 * @author David B. Bracewell
 */
public interface VectorStore<KEY> extends Iterable<LabeledVector>, AutoCloseable, Closeable, Commitable {

   /**
    * The number of vectors stored
    *
    * @return the number of vectors
    */
   int size();

   /**
    * The dimension of the vectors in the store
    *
    * @return the dimension of the vectors
    */
   int dimension();

   /**
    * Adds a vector to the store
    *
    * @param vector the vector to add
    */
   void add(LabeledVector vector);


   /**
    * Adds a vector to the store associating it with the given key
    *
    * @param key    The key to associate the vector with
    * @param vector The vector
    */
   default void add(@NonNull KEY key, @NonNull Vector vector) {
      add(new LabeledVector(key, vector));
   }

   /**
    * Queries the vector store for the nearest vectors to the given <code>query</code> vector returning only matches
    * whose score pass the given <code>threshold</code>. How the threshold is used is determined by the type of measure
    * used in the vector store.
    *
    * @param query     the query vector
    * @param threshold the threshold to filter vectors
    * @return the list of vectors with their labels and scored by the stores measure with respect to the query vector.
    */
   List<ScoredLabelVector> nearest(Vector query, double threshold);


   /**
    * Queries the vector store for the nearest vectors to the given <code>query</code> vector returning only the top
    * <code>K</code> matches whose score pass the given <code>threshold</code>. How the threshold is used is determined
    * by the type of measure used in the vector store.
    *
    * @param query     the query vector
    * @param K         the maximum number of results to return
    * @param threshold the threshold to filter vectors
    * @return the list of vectors with their labels and scored by the stores measure with respect to the query vector.
    */
   List<ScoredLabelVector> nearest(@NonNull Vector query, int K, double threshold);

   /**
    * Queries the vector store for the nearest vectors to the given <code>query</code> vector.
    *
    * @param query the query vector
    * @return the list of vectors with their labels and scored by the stores measure with respect to the query vector.
    */
   List<ScoredLabelVector> nearest(Vector query);


   /**
    * Queries the vector store for the nearest vectors to the given <code>query</code> vector returning only the top
    * <code>K</code>.
    *
    * @param query the query vector
    * @param K     the maximum number of results to return
    * @return the list of vectors with their labels and scored by the stores measure with respect to the query vector.
    */
   List<ScoredLabelVector> nearest(@NonNull Vector query, int K);


   /**
    * The label keys in the store
    *
    * @return the set of vector label keys
    */
   Set<KEY> keySet();

   /**
    * Gets the vector associated with the given key.
    *
    * @param key the key to look up
    * @return the labeled vector or null if key is not in store
    */
   LabeledVector get(KEY key);

   /**
    * Determines if a vector with the label of the given key is in the store.
    *
    * @param key the key
    * @return True if a vector is associated with the given key, False otherwise
    */
   boolean containsKey(KEY key);

   /**
    * Removes the given vector
    *
    * @param vector the vector to remove
    * @return True if removed, False otherwise
    */
   boolean remove(LabeledVector vector);

   @Override
   default void commit() {

   }

   @Override
   default void close() throws IOException {

   }
}// END OF VectorStore
