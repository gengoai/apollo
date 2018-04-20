package com.gengoai.apollo.ml.data.source;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.sequence.Sequence;
import com.gengoai.function.Unchecked;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.apollo.ml.Example;
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
