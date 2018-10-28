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

import com.gengoai.NamedParameters;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.hash.LocalitySensitiveHash;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.json.Json;

import java.io.IOException;
import java.io.Serializable;
import java.util.Iterator;
import java.util.Set;
import java.util.stream.Stream;

import static com.gengoai.NamedParameters.params;

/**
 * <p>Abstract base interface for LSH based NDArray stores.</p>
 *
 * @author David B. Bracewell
 */
public class LSHVectorStore implements VectorStore, Serializable {
   public static final String LSH_EXT = ".lsh.json.gz";
   private final LocalitySensitiveHash<String> lsh;
   private final VectorStore store;

   private LSHVectorStore(LocalitySensitiveHash<String> lsh, VectorStore store) {
      this.lsh = lsh;
      this.store = store;
   }

   @Override
   public boolean containsKey(String String) {
      return store.containsKey(toString());
   }

   @Override
   public int dimension() {
      return store.dimension();
   }

   @Override
   public NDArray get(String term) {
      return store.get(term);
   }

   @Override
   public Iterator<NDArray> iterator() {
      return store.iterator();
   }

   @Override
   public Set<String> keySet() {
      return store.keySet();
   }

   @Override
   public void write(Resource location) throws IOException {
      Resource lshLoc = Resources.fromFile(location.path() + LSH_EXT).setIsCompressed(true);
      Json.dump(lsh, lshLoc);
      store.write(location);
   }

   @Override
   public int size() {
      return store.size();
   }

   @Override
   public NamedParameters<VSParameter> getParameters() {
      return params(VSParameter.LSH, lsh.getParameters())
                .setAll(store.getParameters().asMap());
   }

   @Override
   public Stream<NDArray> query(VSQuery query) {
      NDArray queryVector = query.queryVector(this);
      return query.applyFilters(lsh.query(queryVector)
                                   .stream()
                                   .map(k -> {
                                      NDArray v = store.get(k);
                                      v.setWeight(lsh.getMeasure().calculate(queryVector, v));
                                      return v;
                                   }));
   }

   public static class Builder implements VSBuilder {
      private LocalitySensitiveHash<String> lsh;
      private final NamedParameters<VSParameter> parameters;
      private final VSBuilder builder;

      public Builder(VSBuilder builder, NamedParameters<VSParameter> parameters) {
         this.builder = builder;
         this.parameters = parameters;
         this.lsh = new LocalitySensitiveHash<>(parameters.get(VSParameter.LSH));
      }

      @Override
      public VSBuilder add(String key, NDArray vector) {
         builder.add(key, vector);
         lsh.index(key, vector);
         return this;
      }

      @Override
      public VectorStore build() {
         VectorStore vs = builder.build();

         if (!parameters.getBoolean(VSParameter.IN_MEMORY)) {
            Resource lshLoc = Resources.fromFile(parameters.getString(VSParameter.LOCATION) + LSH_EXT)
                                       .setIsCompressed(true);

            boolean isLoading = lsh.size() == 0;
            try {
               if (isLoading) {
                  if (lshLoc.exists()) {
                     lsh = LocalitySensitiveHash.fromJson(Json.parse(lshLoc), String.class);
                  } else {
                     vs.forEach(n -> lsh.index(n.getLabel(), n));
                     Json.dump(lsh, lshLoc);
                  }
               } else {
                  Json.dump(lsh, lshLoc);
               }
            } catch (IOException e) {
               throw new RuntimeException(e);
            }
         }

         return new LSHVectorStore(lsh, vs);
      }
   }

}

//}// END OF LSHVectorStore
