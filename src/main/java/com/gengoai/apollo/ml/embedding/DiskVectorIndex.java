/*
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

package com.gengoai.apollo.ml.embedding;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.collection.disk.NavigableDiskMap;
import com.gengoai.io.resource.Resource;
import lombok.NonNull;

import java.io.Serializable;
import java.util.stream.Stream;

public class DiskVectorIndex implements VectorIndex, Serializable {
   private static final long serialVersionUID = 1L;
   private final int dimension;
   private final int size;
   private final NavigableDiskMap<Integer, NDArray> store;

   public DiskVectorIndex(@NonNull Resource file, String namespace) {
      this.store = NavigableDiskMap.<Integer, NDArray>builder()
            .compressed(true)
            .readOnly(true)
            .namespace(namespace)
            .file(file)
            .build();
      this.size = this.store.size();
      this.dimension = store.getHandle().getInteger("dimension").intValue();
   }

   public static DiskVectorIndex create(@NonNull NDArray[] vectors, @NonNull Resource file, String namespace) {
      NavigableDiskMap<Integer, NDArray> store = NavigableDiskMap.<Integer, NDArray>builder()
            .compressed(true)
            .namespace(namespace)
            .file(file)
            .build();
      store.getHandle().getInteger("dimension").set((int) vectors[0].length());
      for(int i = 0; i < vectors.length; i++) {
         store.put(i, vectors[i]);
      }
      store.commit();
      return new DiskVectorIndex(file, namespace);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray lookup(int index) {
      return store.get(index);
   }

   @Override
   public int size() {
      return size;
   }

   @Override
   public Stream<NDArray> stream() {
      return store.values().stream();
   }

}//END OF DiskVectorIndex
