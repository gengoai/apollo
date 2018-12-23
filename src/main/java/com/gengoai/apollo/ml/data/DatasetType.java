package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;

import java.io.IOException;

/**
 * Defines how the dataset is stored/processed.
 *
 * @author David B. Bracewell
 */
public enum DatasetType {
   /**
    * Distributed using Apache Spark
    */
   Distributed {
      @Override
      public StreamingContext getStreamingContext() {
         return StreamingContext.distributed();
      }

      @Override
      public Dataset createDataset(MStream<Example> examples) {
         return new DistributedDataset(examples);
      }

      @Override
      public Dataset loadDataset(Resource location, DataSource dataSource) throws IOException {
         return createDataset(dataSource.stream(location, true));
      }
   },
   /**
    * All data is stored in-memory on local machine.
    */
   InMemory {
      @Override
      public Dataset createDataset(MStream<Example> examples) {
         return new InMemoryDataset(examples.collect());
      }


      @Override
      public Dataset loadDataset(Resource location, DataSource dataSource) throws IOException {
         return createDataset(dataSource.stream(location, false));
      }
   },
   /**
    * Data is stored on disk
    */
   OffHeap {
      @Override
      public Dataset createDataset(MStream<Example> examples) {
         return null;
      }

      @Override
      public Dataset loadDataset(Resource location, DataSource dataSource) throws IOException {
         return null;
      }
   },
   /**
    * Data is stored in a Mango stream, what kind may not be known.
    */
   Stream {
      @Override
      public Dataset createDataset(MStream<Example> examples) {
         return null;
      }

      @Override
      public Dataset loadDataset(Resource location, DataSource dataSource) throws IOException {
         return null;
      }
   };


   /**
    * Gets the streaming context.
    *
    * @return the streaming context
    */
   public StreamingContext getStreamingContext() {
      return StreamingContext.local();
   }


   /**
    * Of dataset.
    *
    * @param examples the examples
    * @return the dataset
    */
   public abstract Dataset createDataset(MStream<Example> examples);


   /**
    * Load dataset.
    *
    * @param location the location
    * @return the dataset
    * @throws IOException the io exception
    */
   public final Dataset loadDataset(Resource location) throws IOException {
      return loadDataset(location, new JsonDataSource());
   }

   /**
    * Load dataset.
    *
    * @param location   the location
    * @param dataSource the data source
    * @return the dataset
    * @throws IOException the io exception
    */
   public abstract Dataset loadDataset(Resource location, DataSource dataSource) throws IOException;


}//END OF DatasetType
