package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.io.IOException;

/**
 * <p>Reads individual sequences stored one json per line using the {@link Example#toJson()}</p>
 *
 * @author David B. Bracewell
 */
public class JsonSequenceDataSource extends DataSource<Sequence> {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Json Sequence Data source.
    *
    * @param resource the resource
    */
   public JsonSequenceDataSource(@NonNull Resource resource) {
      super(resource);
   }

   @Override
   public MStream<Sequence> stream() throws IOException {
      return getStreamingContext().textFile(getResource())
                                  .map(Unchecked.function(json -> Example.fromJson(json, Sequence.class)));
   }

}// END OF JsonSequenceDataSource
