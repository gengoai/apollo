package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;

import java.io.IOException;

/**
 * @author David B. Bracewell
 */
public class DatasetBuilder {
   private DatasetType type = DatasetType.InMemory;
   private DataSource dataSource = null;

   public DatasetBuilder type(DatasetType type) {
      this.type = type;
      return this;
   }

   public DatasetBuilder dataSource(DataSource dataSource) {
      this.dataSource = dataSource;
      return this;
   }

   public Dataset source(MStream<Example> stream) {
      return type.create(stream);
   }

   public Dataset source(Resource location) throws IOException {
      if (dataSource == null) {
         return type.read(location);
      }
      return type.read(location, dataSource);
   }


}//END OF DatasetBuilder
