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

package com.gengoai.apollo.ml.data;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.stream.MStream;
import lombok.NonNull;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class StreamBasedVectorizedDataset implements VectorizedDataset {
   private static final long serialVersionUID = 1L;
   private final DatasetType datasetType;
   private MStream<NDArray> stream;

   /**
    * Instantiates a new Base stream dataset.
    *
    * @param datasetType the dataset type
    * @param stream      the stream
    */
   public StreamBasedVectorizedDataset(@NonNull DatasetType datasetType,
                                       MStream<NDArray> stream) {
      this.datasetType = datasetType;
      this.stream = stream == null
                    ? datasetType.getStreamingContext().empty()
                    : stream;
   }

   @Override
   public Iterator<VectorizedDataset> batchIterator(int batchSize) {
      Validation.checkArgument(batchSize > 0, "Batch size must be > 0");
      return stream.partition(batchSize)
                   .map(batch -> datasetType.createVectorizedDataset(datasetType.getStreamingContext()
                                                                                .stream(batch)))
                   .iterator();
   }

   @Override
   public VectorizedDataset cache() {
      stream.cache();
      return this;
   }

   @Override
   public List<NDArray> take(int n) {
      return stream.take(n);
   }

   @Override
   public MStream<NDArray> stream() {
      return stream;
   }

   @Override
   public VectorizedDataset slice(long start, long end) {
      return null;
   }

   @Override
   public long size() {
      return stream.count();
   }

   @Override
   public VectorizedDataset shuffle(@NonNull Random random) {
      return datasetType.createVectorizedDataset(stream.shuffle(random));
   }

   @Override
   public DatasetType getType() {
      return datasetType;
   }

   @Override
   public void close() throws Exception {
      stream.close();
   }

   @Override
   public NDArray get(long index) {
      return stream.skip(index).first().orElse(null);
   }

   @Override
   public Iterator<NDArray> iterator() {
      return stream.iterator();
   }

   @Override
   public MStream<NDArray> parallelStream() {
      return stream.parallel();
   }
}//END OF StreamBasedVectorizedDataset
