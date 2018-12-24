package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;

import java.io.IOException;

/**
 * @author David B. Bracewell
 */
public interface DataSource {

   MStream<Example> stream(Resource location) throws IOException;

}//END OF DataSource
