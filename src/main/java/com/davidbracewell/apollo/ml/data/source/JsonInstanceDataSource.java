package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;

import java.io.IOException;

/**
 * @author David B. Bracewell
 */
public class JsonInstanceDataSource extends DataSource<Instance> {
   private static final long serialVersionUID = -5256453124141512217L;

   /**
   * Instantiates a new Data source.
   *
   * @param resource the resource
   */
  public JsonInstanceDataSource(Resource resource) {
    super(resource);
  }

  @Override
  public MStream<Instance> stream() throws IOException {
    return getStreamingContext().textFile(getResource().path())
      .map(Unchecked.function(json -> Example.fromJson(json, Instance.class)));
  }

}// END OF JsonInstanceDataSource
