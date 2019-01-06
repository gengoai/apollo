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

package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.stream.MStream;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * <p>
 * A Dataset implementation that keeps all examples in memory on a single machine.
 * </p>
 *
 * @author David B. Bracewell
 */
public class InMemoryDataset extends Dataset {
   private static final long serialVersionUID = 1L;
   private final List<Example> examples;

   /**
    * Instantiates a new In memory dataset.
    *
    * @param examples the examples
    */
   public InMemoryDataset(Collection<Example> examples) {
      super(DatasetType.InMemory);
      this.examples = new ArrayList<>(examples);
   }

   @Override
   protected void addAll(MStream<Example> stream) {
      stream.forEachLocal(examples::add);
   }

   @Override
   public Dataset cache() {
      return this;
   }

   @Override
   public void close() throws Exception {

   }

   @Override
   public Iterator<Example> iterator() {
      return examples.iterator();
   }

   @Override
   public int size() {
      return examples.size();
   }


   @Override
   public MStream<Example> stream() {
      return getStreamingContext().stream(examples);
   }

}//END OF InMemoryDataset
