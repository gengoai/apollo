package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.stream.StreamingContext;

/**
 * The dataset type.
 */
public enum DatasetType {
   /**
    * Distributed type.
    */
   Distributed {
      @Override
      public StreamingContext getStreamingContext() {
         return StreamingContext.distributed();
      }
   },
   /**
    * In memory type.
    */
   InMemory,
   /**
    * Off heap type.
    */
   OffHeap,
   /**
    * Stream dataset type.
    */
   Stream;


   /**
    * Gets streaming context.
    *
    * @return the streaming context
    */
   public StreamingContext getStreamingContext() {
      return StreamingContext.local();
   }


}//END OF DatasetType
