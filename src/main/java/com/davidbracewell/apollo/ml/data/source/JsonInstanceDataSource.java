package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.io.IOException;

/**
 * <p>Reads individual instances stored one json per line using the {@link Example#toJson()}</p>
 *
 * @author David B. Bracewell
 */
public class JsonInstanceDataSource extends DataSource<Instance> {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Json Instance data source.
    *
    * @param resource the resource to read from
    */
   public JsonInstanceDataSource(@NonNull Resource resource) {
      super(resource);
   }

   @Override
   public MStream<Instance> stream() throws IOException {
      return getStreamingContext().textFile(getResource())
                                  .map(Unchecked.function(json -> Example.fromJson(json, Instance.class)));
   }

}// END OF JsonInstanceDataSource
