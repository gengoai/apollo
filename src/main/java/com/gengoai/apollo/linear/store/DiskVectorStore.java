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
import com.gengoai.conversion.Cast;
import com.gengoai.io.resource.Resource;
import org.h2.mvstore.MVMap;
import org.h2.mvstore.MVStore;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class DiskVectorStore implements VectorStore {
   private File dbFile;
   private volatile transient MVStore store;
   private volatile transient MVMap<String, NDArray> map;


   @Override
   public boolean containsKey(String key) {
      return map.containsKey(key);
   }

   @Override
   public int dimension() {
      return Cast.<Integer>as(store.openMap("vs_meta").get("dimension"));
   }

   @Override
   public NDArray get(String key) {
      NDArray toReturn = map.get(key);
      if (toReturn == null) {
         toReturn = NDArrayFactory.SPARSE.zeros(dimension());
      }
      return toReturn;
   }

   @Override
   public VectorStoreParameter getParameters() {
      return null;
   }

   @Override
   public Iterator<NDArray> iterator() {
      return map.values().iterator();
   }

   @Override
   public Set<String> keySet() {
      return map.keySet();
   }

   @Override
   public int size() {
      return map.size();
   }

   @Override
   public void write(Resource location) throws IOException {

   }
}//END OF DiskVectorStore
