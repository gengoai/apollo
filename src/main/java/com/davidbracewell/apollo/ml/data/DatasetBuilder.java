package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.apollo.ml.*;
import com.davidbracewell.apollo.ml.data.source.DataSource;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.guava.common.base.Throwables;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.NonNull;

import java.io.IOException;
import java.util.Collection;
import java.util.stream.Stream;

/**
 * <p>Builder for {@link Dataset}</p>
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
public class DatasetBuilder<T extends Example> {
   private final LabelEncoder labelEncoder;
   private final Class<T> exampleType;
   private DataSource<T> dataSource;
   private DatasetType type = DatasetType.InMemory;
   private Encoder featureEncoder = new IndexEncoder();
   private MStream<T> source;
   private Resource load;
   private Vectorizer vectorizer = new DefaultVectorizer();

   /**
    * Instantiates a new Dataset builder.
    *
    * @param labelEncoder the label encoder
    * @param exampleType  the example type
    */
   protected DatasetBuilder(@NonNull LabelEncoder labelEncoder, @NonNull Class<T> exampleType) {
      this.labelEncoder = labelEncoder;
      this.exampleType = exampleType;
   }


   private Dataset<T> createDataset() {
      switch (type) {
         case Distributed:
            return new DistributedDataset<>(featureEncoder, labelEncoder, PreprocessorList.empty(), vectorizer);
         case OffHeap:
            return new OffHeapDataset<>(featureEncoder, labelEncoder, PreprocessorList.empty(), vectorizer);
         default:
            return new InMemoryDataset<>(featureEncoder, labelEncoder, PreprocessorList.empty(), vectorizer);
      }
   }

   /**
    * Sets the feature encoder to use.
    *
    * @param featureEncoder the feature encoder
    */
   public DatasetBuilder<T> featureEncoder(@NonNull Encoder featureEncoder) {
      this.featureEncoder = featureEncoder;
      return this;
   }

   /**
    * Sets the feature encoder to use.
    *
    * @param datasetFile the feature encoder
    */
   public Dataset<T> load(@NonNull Resource datasetFile) {
      try {
         return createDataset().read(datasetFile, exampleType);
      } catch (IOException e) {
         throw Throwables.propagate(e);
      }
   }

   /**
    * Sets the streaming source from a Java Stream.
    *
    * @param stream the stream
    * @return the dataset builder
    */
   public Dataset<T> source(@NonNull Stream<T> stream) {
      Dataset<T> dataset = createDataset();
      dataset.addAll(StreamingContext.local().stream(stream));
      return dataset;
   }

   /**
    * Sets the streaming source from a collection of examples.
    *
    * @param collection the collection of examples
    * @return the dataset builder
    */
   public Dataset<T> source(@NonNull Collection<T> collection) {
      Dataset<T> dataset = createDataset();
      dataset.addAll(StreamingContext.local().stream(collection));
      return dataset;
   }

   /**
    * Sets the examples to be read in from the given data source.
    *
    * @param dataSource the data source
    * @return the dataset builder
    */
   public Dataset<T> source(@NonNull DataSource<T> dataSource) {
      Dataset<T> dataset = createDataset();
      dataSource.setStreamingContext(type.getStreamingContext());
      try {
         dataset.addAll(dataSource.stream());
      } catch (IOException e) {
         throw Throwables.propagate(e);
      }
      return dataset;
   }

   /**
    * Sets the streaming source from a Mango Stream.
    *
    * @param stream the stream
    * @return the dataset builder
    */
   public Dataset<T> source(@NonNull MStream<T> stream) {
      Dataset<T> dataset = createDataset();
      dataset.addAll(stream);
      return dataset;
   }

   /**
    * Sets the type (In-Memory, Distributed, or Off Heap) of the dataset
    *
    * @param type the type
    * @return the dataset builder
    */
   public DatasetBuilder<T> type(@NonNull DatasetType type) {
      this.type = type;
      return this;
   }

   public DatasetBuilder<T> vectorizer(@NonNull Vectorizer vectorizer) {
      this.vectorizer = vectorizer;
      return this;
   }

}// END OF DatasetBuilder
