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
import lombok.NonNull;
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap;

import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * <p>Implementation of an LSH vector store in which vectors are stored in memory.</p>
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
public class InMemoryLSHVectorStore<KEY> extends LSHVectorStore<KEY> {
   private static final long serialVersionUID = 1L;
   private final AtomicInteger vectorIDGenerator = new AtomicInteger();
   private final IntObjectHashMap<LabeledVector> vectorIDMap = new IntObjectHashMap<>();
   private final ObjectIntHashMap<KEY> keys = new ObjectIntHashMap<>();

   /**
    * Instantiates a new in-memory LSH vector store
    *
    * @param lsh the <code>InMemoryLSH</code> to use to build the vector store.
    */
   public InMemoryLSHVectorStore(@NonNull InMemoryLSH lsh) {
      super(lsh);
   }

   public VectorStore<KEY> createNew() {
      return InMemoryLSH.<KEY>builder()
                .dimension(this.dimension())
                .bands(this.lsh.getBands())
                .buckets(this.lsh.getBuckets())
                .signatureFunction(this.lsh.getSignatureFunction())
                .createVectorStore();
   }

   @Override
   public Set<KEY> keySet() {
      return new HashSet<>(keys.keySet());
   }

   @Override
   public boolean containsKey(KEY key) {
      return keySet().contains(key);
   }

   @Override
   protected void removeVector(LabeledVector vector, int id) {
      vectorIDMap.remove(id);
      keys.remove(vector.getLabel());
   }

   @Override
   public Iterator<LabeledVector> iterator() {
      return Collections.unmodifiableCollection(vectorIDMap.values()).iterator();
   }

   @Override
   protected void registerVector(LabeledVector vector, int id) {
      keys.put(vector.getLabel(), id);
      vectorIDMap.put(id, vector);
   }

   @Override
   protected int nextUniqueID() {
      return vectorIDGenerator.getAndIncrement();
   }

   @Override
   protected int getID(KEY key) {
      return keys.get(key);
   }

   @Override
   protected LabeledVector getVectorByID(int id) {
      return vectorIDMap.get(id);
   }

   @Override
   public int size() {
      return vectorIDMap.size();
   }

}// END OF LSHVectorStore
