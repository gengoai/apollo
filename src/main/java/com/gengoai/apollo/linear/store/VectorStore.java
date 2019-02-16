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
 *
 */

package com.gengoai.apollo.linear.store;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.VectorComposition;
import com.gengoai.collection.Streams;
import com.gengoai.io.IndexedFile;
import com.gengoai.io.resource.Resource;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * <p>A mapping of strings into vector representations. </p>
 *
 * @author David B. Bracewell
 */
public interface VectorStore extends Iterable<NDArray> {

   /**
    * Creates a builder to construct a new {@link VectorStore}
    *
    * @return the vector store builder
    */
   static VSBuilder builder() {
      return builder(new VectorStoreParameter());
   }

   /**
    * Creates a builder to construct a new {@link VectorStore}
    *
    * @param updater consumer to update the {@link VectorStoreParameter}s used to create the builder
    * @return the vector store builder
    */
   static VSBuilder builder(Consumer<VectorStoreParameter> updater) {
      VectorStoreParameter parameter = new VectorStoreParameter();
      updater.accept(parameter);
      return builder(parameter);
   }

   /**
    * Creates a builder to construct a new {@link VectorStore}
    *
    * @param parameter the {@link VectorStoreParameter}s used to create the builder
    * @return the vector store builder
    */
   static VSBuilder builder(VectorStoreParameter parameter) {
      return parameter.type.builder(parameter);
   }

   /**
    * Read vector store.
    *
    * @param vectors the vectors
    * @return the vector store
    * @throws IOException the io exception
    */
   static VectorStore read(Resource vectors) throws IOException {
      File vectorFile = vectors.asFile().orElseThrow(IOException::new);
      File indexFile = IndexedFile.indexFileFor(vectorFile);
      VectorStoreParameter parameter = new VectorStoreParameter();
      parameter.location = vectors;
      if (indexFile.exists()) {
         parameter.type = VectorStoreType.IndexedFile;
      } else {
         parameter.type = VectorStoreType.InMemory;
      }
      return builder(parameter).build();
   }

   /**
    * Creates a vector using the given vector composition for the given words.
    *
    * @param composition the composition function to use
    * @param words       the words whose vectors we want to compose
    * @return a composite vector consisting of the given words and calculated using the given vector composition
    */
   default NDArray compose(VectorComposition composition, String... words) {
      if (words == null) {
         return NDArrayFactory.DEFAULT().zeros(dimension());
      } else if (words.length == 1) {
         return get(words[0]);
      }
      return composition.compose(Arrays.stream(words)
                                       .map(this::get)
                                       .collect(Collectors.toList()));
   }

   /**
    * Determines if a vector with the label of the given String is in the store.
    *
    * @param String the String
    * @return True if a vector is associated with the given String, False otherwise
    */
   boolean containsKey(String String);

   /**
    * The dimension of the vectors in the store
    *
    * @return the dimension of the vectors
    */
   int dimension();

   /**
    * Gets the vector associated with the given String.
    *
    * @param String the String to look up
    * @return the labeled vector or null if String is not in store
    */
   NDArray get(String String);

   /**
    * Create new vector store.
    *
    * @return the vector store
    */
   VectorStoreParameter getParameters();

   /**
    * The label Strings in the store
    *
    * @return the set of vector label Strings
    */
   Set<String> keySet();

   /**
    * Queries the vector store to find similar vectors to the given {@link VSQuery}.
    *
    * @param query the query to use find similar vectors
    * @return Stream of vectors matching the query
    */
   default Stream<NDArray> query(VSQuery query) {
      NDArray queryVector = query.queryVector(this);
      return query.applyFilters(Streams.asParallelStream(iterator())
                                       .map(v -> v.copy().setWeight(query.measure().calculate(v, queryVector))));
   }

   /**
    * Size int.
    *
    * @return the number of vectors
    */
   int size();

   /**
    * Write.
    *
    * @param location the location
    * @throws IOException the io exception
    */
   void write(Resource location) throws IOException;

}// END OF VectorStore
