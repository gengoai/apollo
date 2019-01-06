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

import java.util.Iterator;

/**
 * <p>
 * Abstract base {@link Dataset} backed by an <code>MStream</code>.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class BaseStreamDataset extends Dataset {
   private static final long serialVersionUID = 1L;
   /**
    * The stream of examples
    */
   protected MStream<Example> stream;

   /**
    * Instantiates a new Base stream dataset.
    *
    * @param datasetType the dataset type
    * @param stream      the stream
    */
   public BaseStreamDataset(DatasetType datasetType, MStream<Example> stream) {
      super(datasetType);
      this.stream = stream == null
                    ? datasetType.getStreamingContext().empty()
                    : stream;
   }

   @Override
   protected void addAll(MStream<Example> stream) {
      this.stream = this.stream.union(stream);
   }

   @Override
   public Dataset cache() {
      if (getStreamingContext().isDistributed()) {
         return newDataset(stream.cache());
      }
      return new InMemoryDataset(stream.collect());
   }

   @Override
   public void close() throws Exception {
      stream.close();
   }

   @Override
   public Iterator<Example> iterator() {
      return stream.iterator();
   }


   @Override
   public MStream<Example> stream() {
      return stream;
   }
}//END OF BaseStreamDataset
