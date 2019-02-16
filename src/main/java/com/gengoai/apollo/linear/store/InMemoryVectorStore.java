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
import com.gengoai.collection.Iterators;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.string.Strings;

import java.io.*;
import java.util.*;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.notNullOrBlank;
import static com.gengoai.apollo.linear.store.VSTextUtils.vectorToLine;

/**
 * The type Default vector store.
 *
 * @author David B. Bracewell
 */
public class InMemoryVectorStore implements VectorStore, Serializable {
   private static final long serialVersionUID = 1L;
   private final int dimension;
   private final Map<String, NDArray> vectorMap = new HashMap<>();

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
   public VectorStoreParameter getParameters() {
      VectorStoreParameter parameter = new VectorStoreParameter();
      parameter.type = VectorStoreType.InMemory;
      return parameter;
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
   public void write(Resource location) throws IOException {
      try (BufferedWriter writer = new BufferedWriter(location.writer())) {
         for (Map.Entry<String, NDArray> entry : vectorMap.entrySet()) {
            writer.write(entry.getKey());
            writer.write(" ");
            writer.write(vectorToLine(entry.getValue()));
            writer.write("\n");
         }
      }
   }


   /**
    * The type Builder.
    */
   public static class Builder implements VSBuilder {
      private final VectorStoreParameter params;
      private final Map<String, NDArray> vectors = new HashMap<>();
      private int dimension = -1;

      public Builder(VectorStoreParameter params) {
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
         InMemoryVectorStore vs;
         if (vectors.size() > 0 || params.location == null) {
            vs = new InMemoryVectorStore(dimension);
            vs.vectorMap.putAll(vectors);
         } else {
            File vectorFile = params.location.asFile().orElseThrow(IllegalArgumentException::new);
            try {
               dimension = VSTextUtils.determineDimension(vectorFile);
               vs = new InMemoryVectorStore(dimension);
               try (BufferedReader reader = new BufferedReader(Resources.fromFile(vectorFile).reader())) {
                  String line;
                  while ((line = reader.readLine()) != null) {
                     if (Strings.isNotNullOrBlank(line)) {
                        NDArray v = VSTextUtils.convertLineToVector(line, dimension);
                        vs.vectorMap.put(v.getLabel(), v);
                     }
                  }
               }


            } catch (IOException e) {
               throw new RuntimeException(e.getCause());
            }
         }


         return vs;
      }

   }// END OF Builder

}//END OF InMemoryVectorStore
