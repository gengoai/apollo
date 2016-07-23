package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;

import java.io.IOException;

/**
 * @author David B. Bracewell
 */
public class JsonSequenceDataSource extends DataSource<Sequence> {
  /**
   * Instantiates a new Data source.
   *
   * @param resource the resource
   */
  public JsonSequenceDataSource(Resource resource) {
    super(resource);
  }

  @Override
  public MStream<Sequence> stream() throws IOException {
    return getStreamingContext().textFile(getResource().path())
      .map(Unchecked.function(json -> Example.fromJson(json, Sequence.class)));
  }

}// END OF JsonSequenceDataSource
