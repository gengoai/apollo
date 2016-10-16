package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.LabelEncoder;
import com.davidbracewell.apollo.ml.data.source.DataSource;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import com.google.common.base.Throwables;
import lombok.NonNull;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.io.IOException;
import java.util.Collection;
import java.util.stream.Stream;

/**
 * <p>Builder for datasets</p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public class DatasetBuilder<T extends Example> {
   private final LabelEncoder labelEncoder;
   private final Class<T> exampleType;
   @Setter(onParam = @_({@NonNull}))
   private DataSource<T> dataSource;
   @Setter(onParam = @_({@NonNull}))
   private DatasetType type = DatasetType.InMemory;
   @Setter(onParam = @_({@NonNull}))
   private Encoder featureEncoder = new IndexEncoder();
   @Setter(onParam = @_({@NonNull}))
   private MStream<T> source;
   @Setter(onParam = @_({@NonNull}))
   private Resource load;

   protected DatasetBuilder(@NonNull LabelEncoder labelEncoder, @NonNull Class<T> exampleType) {
      this.labelEncoder = labelEncoder;
      this.exampleType = exampleType;
   }

   /**
    * Sets the streaming source from a Java Stream.
    *
    * @param stream the stream
    * @return the dataset builder
    */
   public DatasetBuilder<T> localSource(@NonNull Stream<T> stream) {
      this.source = StreamingContext.local().stream(stream);
      return this;
   }

   public DatasetBuilder<T> data(@NonNull Collection<T> collection) {
      this.source = StreamingContext.local().stream(collection);
      return this;
   }

   /**
    * Builds the dataset using the provided values.
    *
    * @return the dataset
    */
   public Dataset<T> build() {
      Dataset<T> dataset;
      switch (type) {
         case Distributed:
            dataset = new DistributedDataset<>(featureEncoder, labelEncoder, PreprocessorList.empty());
            break;
         case OffHeap:
            dataset = new OffHeapDataset<>(featureEncoder, labelEncoder, PreprocessorList.empty());
            break;
         default:
            dataset = new InMemoryDataset<>(featureEncoder, labelEncoder, PreprocessorList.empty());
      }


      if (source != null) {
         dataset.addAll(source);
      }

      if (dataSource != null) {
         dataSource.setStreamingContext(type.getStreamingContext());
         try {
            dataset.addAll(dataSource.stream());
         } catch (IOException e) {
            throw Throwables.propagate(e);
         }
      } else if (load != null) {
         try {
            dataset.read(load, exampleType);
         } catch (IOException e) {
            throw Throwables.propagate(e);
         }
      }

      return dataset;
   }

}// END OF DatasetBuilder
