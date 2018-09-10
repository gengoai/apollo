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
import com.gengoai.apollo.linear.hash.LocalitySensitiveHash;
import com.gengoai.io.resource.Resource;

import java.io.IOException;
import java.io.Serializable;
import java.util.Iterator;
import java.util.Set;

import static com.gengoai.Parameters.params;

/**
 * <p>Abstract base interface for LSH based NDArray stores.</p>
 *
 * @author David B. Bracewell
 */
public class LSHVectorStore implements VectorStore, Serializable {
   private LocalitySensitiveHash lsh;
   private VectorStore store;

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

   }

   @Override
   public int size() {
      return store.size();
   }

   @Override
   public Parameters<VSParams> getParameters() {
      return params(VSParams.LSH, lsh.getParameters())
                .setAll(store.getParameters().asMap());
   }

}

//}// END OF LSHVectorStore
