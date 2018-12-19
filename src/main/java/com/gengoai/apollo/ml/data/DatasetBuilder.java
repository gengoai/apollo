package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;

/**
 * @author David B. Bracewell
 */
public class DatasetBuilder {
   private DatasetType datasetType = DatasetType.InMemory;
   private MStream<Example> source;
   private Resource fileLocation;
   private DataSource dataSource;


   public static DatasetBuilder create(){
      return new DatasetBuilder();
   }

   protected DatasetBuilder(){

   }




   public DatasetType getDatasetType() {
      return datasetType;
   }

   public DatasetBuilder setDatasetType(DatasetType datasetType) {
      this.datasetType = datasetType;
      return this;
   }

}//END OF DatasetBuilder
