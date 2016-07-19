package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;

/**
 * The interface Data format.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public abstract class DataSource<T extends Example> implements Serializable {

  private final Resource resource;
  private StreamingContext streamingContext = StreamingContext.local();

  /**
   * Instantiates a new Data source.
   *
   * @param resource the resource
   */
  public DataSource(Resource resource) {
    this.resource = resource;
  }


  /**
   * Gets resource.
   *
   * @return the resource
   */
  public Resource getResource() {
    return resource;
  }

  /**
   * Gets streaming context.
   *
   * @return the streaming context
   */
  public StreamingContext getStreamingContext() {
    return streamingContext;
  }

  /**
   * Sets streaming context.
   *
   * @param streamingContext the streaming context
   */
  public void setStreamingContext(@NonNull StreamingContext streamingContext) {
    this.streamingContext = streamingContext;
  }

  /**
   * Read m stream.
   *
   * @return the m stream
   * @throws IOException the io exception
   */
  public abstract MStream<T> stream() throws IOException;


}//END OF DataProvider
