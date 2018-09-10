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

import com.gengoai.Parameters;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.VectorComposition;
import com.gengoai.collection.Streams;
import com.gengoai.io.resource.Resource;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * <p>A vector store provides access and lookup of vectors by labels and to find vectors in the store closest to query
 * vectors. </p>
 *
 * @author David B. Bracewell
 */
public interface VectorStore extends Iterable<NDArray> {

   /**
    * Creates a vector using the given vector composition for the given words.
    *
    * @param composition the composition function to use
    * @param words       the words whose vectors we want to compose
    * @return a composite vector consisting of the given words and calculated using the given vector composition
    */
   @SuppressWarnings("unchecked")
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
    * The label Strings in the store
    *
    * @return the set of vector label Strings
    */
   Set<String> keySet();

   void write(Resource location) throws IOException;

   /**
    * Queries the vector store to find similar vectors to the given {@link VSQuery}.
    *
    * @param query the query to use find similar vectors
    * @return Stream of vectors matching the query
    */
   default Stream<NDArray> query(VSQuery query) {
      NDArray queryVector = query.queryVector(this);
      Stream<NDArray> stream = Streams.asParallelStream(iterator())
                                      .map(v -> v.copy().setWeight(query.measure().calculate(v, queryVector)));
      if (Double.isFinite(query.threshold())) {
         stream = stream.filter(v -> query.measure().getOptimum().test(v.getWeight(), query.threshold()));
      }
      stream = stream.sorted((v1, v2) -> query.measure().getOptimum().compare(v1.getWeight(), v2.getWeight()));
      Set<String> exclude = query.getExcludedLabels();
      if (exclude.size() > 0) {
         stream = stream.filter(v -> !exclude.contains(v.getLabel()));
      }
      if (query.limit() > 0 && query.limit() < Integer.MAX_VALUE) {
         stream = stream.limit(query.limit());
      }
      return stream;
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
   VSBuilder toBuilder();


   static VectorStore create(Parameters<VSParams> params) {
      VectorStore store;
      if (params.getBoolean(VSParams.IN_MEMORY)) {
         store = new InMemoryVectorStore(params.getInt(VSParams.DIMENSION));
      } else {
         store = new DiskBasedVectorStore(new File(params.getString(VSParams.LOCATION)),
                                          params.getInt(VSParams.CACHE_SIZE));
      }


      return store;
   }


}// END OF VectorStore
