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
import com.gengoai.apollo.linear.VectorComposition;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Streams;
import com.gengoai.conversion.Cast;
import com.gengoai.io.Commitable;
import com.gengoai.tuple.Tuple;
import lombok.NonNull;

import java.io.Closeable;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import static com.gengoai.tuple.Tuples.$;

/**
 * <p>A vector store provides access and lookup of vectors by labels and to find vectors in the store closest to query
 * vectors. </p>
 *
 * @param <KEY> the type of key associated with vectors
 * @author David B. Bracewell
 */
public interface VectorStore<KEY> extends Iterable<NDArray>, AutoCloseable, Closeable, Commitable {

   @Override
   default void close() throws IOException {

   }

   @Override
   default void commit() {

   }

   /**
    * Creates a vector using the given vector composition for the given words.
    *
    * @param composition the composition function to use
    * @param words       the words whose vectors we want to compose
    * @return a composite vector consisting of the given words and calculated using the given vector composition
    */
   @SuppressWarnings("unchecked")
   default NDArray compose(@NonNull VectorComposition composition, KEY... words) {
      if (words == null) {
         return NDArrayFactory.DEFAULT().zeros(dimension());
      } else if (words.length == 1) {
         return get(words[0]);
      }
      List<NDArray> vectors = new ArrayList<>();
      for (KEY w : words) {
         vectors.add(get(w));
      }
      return composition.compose(vectors);
   }

   /**
    * Determines if a vector with the label of the given key is in the store.
    *
    * @param key the key
    * @return True if a vector is associated with the given key, False otherwise
    */
   boolean containsKey(KEY key);

   /**
    * The dimension of the vectors in the store
    *
    * @return the dimension of the vectors
    */
   int dimension();

   /**
    * Gets the vector associated with the given key.
    *
    * @param key the key to look up
    * @return the labeled vector or null if key is not in store
    */
   NDArray get(KEY key);

   /**
    * @return The query measured for nearest neighbor calls
    */
   Measure getQueryMeasure();

   /**
    * The label keys in the store
    *
    * @return the set of vector label keys
    */
   Collection<KEY> keys();

   /**
    * Queries the vector store for the nearest vectors to the given <code>query</code> vector returning only matches
    * whose score pass the given <code>threshold</code>. How the threshold is used is determined by the type of measure
    * used in the vector store.
    *
    * @param query     the query vector
    * @param threshold the threshold to filter vectors
    * @return the list of vectors with their labels and scored by the stores measure with respect to the query vector.
    */
   default List<NDArray> nearest(NDArray query, double threshold) {
      Validation.checkArgument(query.length() == dimension(),
                               "Dimension mismatch, vector store can only store vectors with k of " + dimension());
      return Streams.asParallelStream(iterator())
                    .map(v -> v.copy().setWeight(getQueryMeasure().calculate(v, query)))
                    .filter(s -> getQueryMeasure().getOptimum().test(s.getWeight(), threshold))
                    .collect(Collectors.toList());
   }

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
   default List<NDArray> nearest(@NonNull NDArray query, int K, double threshold) {
      Validation.checkArgument(query.length() == dimension(),
                                  "Dimension mismatch, vector store can only store vectors with k of " + dimension());
      List<NDArray> vectors = Streams.asParallelStream(iterator())
                                     .map(v -> v.copy().setWeight(getQueryMeasure().calculate(v, query)))
                                     .filter(s -> getQueryMeasure().getOptimum().test(s.getWeight(), threshold))
                                     .sorted((s1, s2) -> getQueryMeasure().getOptimum()
                                                                          .compare(s1.getWeight(), s2.getWeight()))
                                     .collect(Collectors.toList());
      return vectors.subList(0, Math.min(K, vectors.size()));
   }

   /**
    * Queries the vector store for the nearest vectors to the given <code>query</code> vector.
    *
    * @param query the query vector
    * @return the list of vectors with their labels and scored by the stores measure with respect to the query vector.
    */
   default List<NDArray> nearest(NDArray query) {
      Validation.checkArgument(query.length() == dimension(),
                                  "Dimension mismatch, vector store can only store vectors with k of " + dimension());
      return Streams.asParallelStream(iterator())
                    .map(v -> v.copy().setWeight(getQueryMeasure().calculate(v, query)))
                    .collect(Collectors.toList());
   }

   /**
    * Queries the vector store for the nearest vectors to the given <code>query</code> vector returning only the top
    * <code>K</code>.
    *
    * @param query the query vector
    * @param K     the maximum number of results to return
    * @return the list of vectors with their labels and scored by the stores measure with respect to the query vector.
    */
   default List<NDArray> nearest(@NonNull NDArray query, int K) {
      Validation.checkArgument(query.length() == dimension(),
                                  "Dimension mismatch, vector store can only store vectors with k of " + dimension());
      List<NDArray> vectors = Streams.asParallelStream(iterator())
                                     .map(v -> v.copy().setWeight(getQueryMeasure().calculate(v, query)))
                                     .sorted((s1, s2) -> getQueryMeasure().getOptimum()
                                                                          .compare(s1.getWeight(), s2.getWeight()))
                                     .collect(Collectors.toList());
      return vectors.subList(0, Math.min(K, vectors.size()));
   }

   /**
    * Finds the closest K vectors to the given word/feature in the embedding
    *
    * @param word the word/feature whose neighbors we want
    * @param K    the maximum number of neighbors to return
    * @return the list of scored K-nearest vectors
    */
   default List<NDArray> nearest(@NonNull KEY word, int K) {
      return nearest(word, K, Double.NEGATIVE_INFINITY);
   }

   /**
    * Finds the closest K vectors to the given word/feature in the embedding
    *
    * @param word      the word/feature whose neighbors we want
    * @param K         the maximum number of neighbors to return
    * @param threshold threshold for selecting vectors
    * @return the list of scored K-nearest vectors
    */
   default List<NDArray> nearest(@NonNull KEY word, int K, double threshold) {
      NDArray v1 = get(word);
      if (v1 == null) {
         return Collections.emptyList();
      }
      List<NDArray> near = nearest(v1, K + 1, threshold)
                              .stream()
                              .filter(slv -> !word.equals(slv.getLabel()))
                              .collect(Collectors.toList());
      return near.subList(0, Math.min(K, near.size()));
   }

   /**
    * Finds the closest K vectors to the given positive tuple of words/features and not near the negative tuple of
    * words/features in the embedding
    *
    * @param positive  a tuple of words/features (the individual vectors are composed using vector addition) whose
    *                  neighbors we want
    * @param negative  a tuple of words/features (the individual vectors are composed using vector addition) subtracted
    *                  from the positive vectors.
    * @param K         the maximum number of neighbors to return
    * @param threshold threshold for selecting vectors
    * @return the list of scored K-nearest vectors
    */
   default List<NDArray> nearest(@NonNull Tuple positive, @NonNull Tuple negative, int K, double threshold) {
      NDArray pVec = NDArrayFactory.DEFAULT().zeros(dimension());
      positive.forEach(word -> pVec.addi(get(Cast.as(word))));
      NDArray nVec = NDArrayFactory.DEFAULT().zeros(dimension());
      negative.forEach(word -> nVec.addi(get(Cast.as(word))));
      Set<String> ignore = new HashSet<>();
      positive.forEach(o -> ignore.add(o.toString()));
      negative.forEach(o -> ignore.add(o.toString()));
      List<NDArray> vectors = nearest(pVec.sub(nVec), K + positive.degree() + negative.degree(),
                                      threshold)
                                 .stream()
                                 .filter(slv -> !ignore.contains(slv.<String>getLabel()))
                                 .collect(Collectors.toList());
      return vectors.subList(0, Math.min(K, vectors.size()));
   }

   /**
    * Finds the closest K vectors to the given tuple of words/features in the embedding
    *
    * @param words a tuple of words/features (the individual vectors are composed using vector addition) whose neighbors
    *              we want
    * @param K     the maximum number of neighbors to return
    * @return the list of scored K-nearest vectors
    */
   default List<NDArray> nearest(@NonNull Tuple words, int K) {
      return nearest(words, $(), K, Double.NEGATIVE_INFINITY);
   }

   /**
    * The number of vectors stored
    *
    * @return the number of vectors
    */
   int size();

   /**
    * Create new vector store.
    *
    * @return the vector store
    */
   VectorStoreBuilder<KEY> toBuilder();

}// END OF VectorStore
