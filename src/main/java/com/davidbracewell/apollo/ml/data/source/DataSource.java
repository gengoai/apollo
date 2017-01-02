package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.Getter;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;

/**
 * <p>Abstract base class for defining a dataset format. Examples include CSV, LibSVM, and json.</p>
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
public abstract class DataSource<T extends Example> implements Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   private final Resource resource;
   @Getter
   private StreamingContext streamingContext = StreamingContext.local();

   /**
    * Instantiates a new Data source.
    *
    * @param resource the resource containing the data
    */
   public DataSource(Resource resource) {
      this.resource = resource;
   }

   /**
    * Sets the streaming context.
    *
    * @param streamingContext the streaming context
    */
   public void setStreamingContext(@NonNull StreamingContext streamingContext) {
      this.streamingContext = streamingContext;
   }

   /**
    * Provides a stream of examples from the data source
    *
    * @return a Mango stream of examples
    * @throws IOException Something went wrong reading the data source
    */
   public abstract MStream<T> stream() throws IOException;


}//END OF DataSource
