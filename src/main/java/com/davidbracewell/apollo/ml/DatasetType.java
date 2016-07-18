package com.davidbracewell.apollo.ml;

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
  Stream;


  public StreamingContext getStreamingContext() {
    return StreamingContext.local();
  }


}//END OF DatasetType
