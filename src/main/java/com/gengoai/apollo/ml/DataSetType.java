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

package com.gengoai.apollo.ml;

import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import lombok.NonNull;

import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Defines how the dataset is stored/processed.
 *
 * @author David B. Bracewell
 */
public enum DataSetType {
   /**
    * Distributed using Apache Spark
    */
   Distributed {
      @Override
      public StreamingContext getStreamingContext() {
         return StreamingContext.distributed();
      }

      @Override
      protected DataSet create(@NonNull MStream<Datum> stream) {
         return new StreamingDataSet(stream.toDistributedStream());
      }

      @Override
      protected DataSet create(@NonNull Stream<Datum> stream) {
         return new StreamingDataSet(StreamingContext.distributed().stream(stream));
      }
   },
   /**
    * All data is stored in-memory on local machine.
    */
   InMemory {
      @Override
      protected DataSet create(@NonNull MStream<Datum> stream) {
         return new InMemoryDataSet(stream.collect());
      }

      @Override
      protected DataSet create(@NonNull Stream<Datum> stream) {
         return new InMemoryDataSet(stream.collect(Collectors.toList()));
      }
   },
   /**
    * Local Streaming-based dataset
    */
   LocalStreaming {
      @Override
      protected DataSet create(@NonNull MStream<Datum> stream) {
         if(stream.isDistributed()) {
            return new StreamingDataSet(StreamingContext.local()
                                                        .stream(stream.collect()));
         }
         return new StreamingDataSet(stream);
      }

      @Override
      protected DataSet create(@NonNull Stream<Datum> stream) {
         return new StreamingDataSet(StreamingContext.local().stream(stream));
      }
   };

   protected abstract DataSet create(@NonNull MStream<Datum> stream);

   protected abstract DataSet create(@NonNull Stream<Datum> stream);

   /**
    * Gets the streaming context.
    *
    * @return the streaming context
    */
   public StreamingContext getStreamingContext() {
      return StreamingContext.local();
   }

}//END OF DataSetType