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
import com.gengoai.apollo.linear.hash.LSHParameter;
import com.gengoai.collection.Iterators;
import com.gengoai.io.resource.Resource;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

import static com.gengoai.Parameters.params;
import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.notNullOrBlank;
import static com.gengoai.apollo.linear.NDArray.vec2String;

/**
 * The type Default vector store.
 *
 * @author David B. Bracewell
 */
public class InMemoryVectorStore implements VectorStore, Serializable {
   private static final long serialVersionUID = 1L;
   private final Map<String, NDArray> vectorMap = new HashMap<>();
   private final int dimension;

   public InMemoryVectorStore(int dimension) {
      this.dimension = dimension;
   }


   @Override
   public boolean containsKey(String key) {
      return vectorMap.containsKey(key);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(String key) {
      return vectorMap.getOrDefault(key, NDArrayFactory.SPARSE.zeros(dimension));
   }

   @Override
   public Iterator<NDArray> iterator() {
      return Iterators.unmodifiableIterator(Iterators.transform(vectorMap.entrySet().iterator(),
                                                                e -> e.getValue().setLabel(e.getKey())));
   }

   @Override
   public Set<String> keySet() {
      return Collections.unmodifiableSet(vectorMap.keySet());
   }

   @Override
   public int size() {
      return vectorMap.size();
   }

   @Override
   public Parameters<VSParameter> getParameters() {
      return params(VSParameter.IN_MEMORY, true);
   }

   @Override
   public void write(Resource location) throws IOException {
      try (BufferedWriter writer = new BufferedWriter(location.writer())) {
         for (Map.Entry<String, NDArray> entry : vectorMap.entrySet()) {
            writer.write(entry.getKey());
            writer.write(" ");
            writer.write(vec2String(entry.getValue()));
            writer.write("\n");
         }
      }
   }


   /**
    * The type Builder.
    */
   public static class Builder implements VSBuilder {
      private final Map<String, NDArray> vectors = new HashMap<>();
      private int dimension = -1;
      private final Parameters<VSParameter> params;

      public Builder(Parameters<VSParameter> params) {
         this.params = params;
      }

      @Override
      public VSBuilder add(String key, NDArray vector) {
         notNullOrBlank(key, "The key must not be null or blank");
         if (dimension == -1) {
            dimension = (int) vector.length();
         }
         checkArgument(dimension == vector.length(),
                       () -> "Dimension mismatch. (" + dimension + ") != (" + vector.length() + ")");
         vectors.put(key, vector);
         return this;
      }

      @Override
      public VectorStore build() {
         Parameters<LSHParameter> lshParameters = params.get(VSParameter.LSH);
         InMemoryVectorStore vs = new InMemoryVectorStore(dimension);
         vs.vectorMap.putAll(vectors);
         if (lshParameters != null) {
            lshParameters.set(LSHParameter.DIMENSION, dimension);
         }
         return vs;
      }

   }// END OF Builder

}//END OF InMemoryVectorStore
