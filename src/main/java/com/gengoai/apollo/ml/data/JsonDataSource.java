package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.Sequence;
import com.gengoai.function.Unchecked;
import com.gengoai.io.resource.Resource;
import com.gengoai.json.Json;
import com.gengoai.json.JsonEntry;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;

import java.io.IOException;

/**
 * The type Json data source.
 *
 * @author David B. Bracewell
 */
public class JsonDataSource implements DataSource {
   private final boolean distributed;

   public JsonDataSource(boolean distributed) {
      this.distributed = distributed;
   }

   @Override
   public MStream<Example> stream(Resource location) throws IOException {
      return StreamingContext.get(distributed)
                             .textFile(location)
                             .map(Unchecked.function(json -> {
                                JsonEntry entry = Json.parse(json);
                                if (entry.hasProperty("sequence")) {
                                   return entry.getAs(Sequence.class);
                                }
                                return entry.getAs(Instance.class);
                             }));
   }


}//END OF JsonDataSource
