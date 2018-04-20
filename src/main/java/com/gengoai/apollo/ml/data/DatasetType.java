package com.gengoai.apollo.ml.data;

import com.gengoai.stream.StreamingContext;

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
   InMemory,
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


}//END OF DatasetType
