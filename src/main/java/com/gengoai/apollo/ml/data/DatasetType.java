package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
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
   },
   /**
    * All data is stored in-memory on local machine.
    */
   InMemory {
      @Override
      public Dataset of(MStream<Example> examples) {
         return new InMemoryDataset(examples.collect());
      }


      @Override
      public Dataset load(Resource location, DataSource dataSource) throws IOException {
         return of(dataSource.stream(location));
      }


   },
   /**
    * Data is stored on disk
    */
   OffHeap,
   /**
    * Data is stored in a Mango stream, what kind may not be known.
    */
   Stream;


   /**
    * Gets the streaming context.
    *
    * @return the streaming context
    */
   public StreamingContext getStreamingContext() {
      return StreamingContext.local();
   }


   public Dataset of(MStream<Example> examples) {
      return new InMemoryDataset(examples.collect());
   }


   public Dataset load(Resource location, DataSource dataSource) throws IOException {
      return null;
   }


}//END OF DatasetType
