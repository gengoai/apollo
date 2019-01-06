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
import com.gengoai.apollo.ml.data.format.DataFormat;
import com.gengoai.apollo.ml.data.format.JsonDataFormat;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;

import java.io.IOException;

/**
 * Defines how the dataset is stored/processed.
 *
 * @author David B. Bracewell
 */
public enum DatasetType {
   /**
    * Distributed using Apache Spark
    */
   Distributed {
      @Override
      public StreamingContext getStreamingContext() {
         return StreamingContext.distributed();
      }

      @Override
      public Dataset create(MStream<Example> examples) {
         return new DistributedDataset(examples);
      }

      @Override
      public Dataset read(Resource location, DataFormat dataFormat) throws IOException {
         return create(dataFormat.read(location));
      }
   },
   /**
    * All data is stored in-memory on local machine.
    */
   InMemory {
      @Override
      public Dataset create(MStream<Example> examples) {
         return new InMemoryDataset(examples.collect());
      }


      @Override
      public Dataset read(Resource location, DataFormat dataFormat) throws IOException {
         return create(dataFormat.read(location));
      }
   },
   /**
    * Data is stored on disk
    */
   OffHeap {
      @Override
      public Dataset create(MStream<Example> examples) {
         return null;
      }

      @Override
      public Dataset read(Resource location, DataFormat dataFormat) throws IOException {
         return null;
      }
   },
   /**
    * Data is stored in a Mango stream, what kind may not be known.
    */
   Stream {
      @Override
      public Dataset create(MStream<Example> examples) {
         return new MStreamDataset(examples);
      }

      @Override
      public Dataset read(Resource location, DataFormat dataFormat) throws IOException {
         return create(dataFormat.read(location));
      }
   };


   /**
    * Gets the streaming context.
    *
    * @return the streaming context
    */
   public StreamingContext getStreamingContext() {
      return StreamingContext.local();
   }


   /**
    * Of dataset.
    *
    * @param examples the examples
    * @return the dataset
    */
   public abstract Dataset create(MStream<Example> examples);


   /**
    * Load dataset.
    *
    * @param location the location
    * @return the dataset
    * @throws IOException the io exception
    */
   public final Dataset read(Resource location) throws IOException {
      return read(location, new JsonDataFormat(getStreamingContext().isDistributed()));
   }

   /**
    * Load dataset.
    *
    * @param location   the location
    * @param dataFormat the data source
    * @return the dataset
    * @throws IOException the io exception
    */
   public abstract Dataset read(Resource location, DataFormat dataFormat) throws IOException;


}//END OF DatasetType
