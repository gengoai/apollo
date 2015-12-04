package com.davidbracewell.apollo.ml;

import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import lombok.NonNull;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.stream.Stream;

/**
 * The type Dataset builder.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public class DatasetBuilder<T extends Example> {
  @Setter(onParam = @_({@NonNull}))
  private Dataset.Type type = Dataset.Type.InMemory;
  @Setter(onParam = @_({@NonNull}))
  private Encoder featureEncoder = new IndexEncoder();
  @Setter(onParam = @_({@NonNull}))
  private Encoder labelEncoder = new IndexEncoder();
  @Setter
  private MStream<T> source;


  public DatasetBuilder<T> source(@NonNull Stream<T> stream) {
    this.source = Streams.of(stream);
    return this;
  }

  /**
   * Build dataset.
   *
   * @return the dataset
   */
  public Dataset<T> build() {
    Dataset<T> dataset;
    switch (type) {
      case Distributed:
      case OffHeap:
        dataset = new OffHeapDataset<>(featureEncoder, labelEncoder, null);
        break;
      default:
        dataset = new InMemoryDataset<>(featureEncoder, labelEncoder, null);
    }

    if (source != null) {
      dataset.addAll(source);
    }

    return dataset;
  }

}// END OF DatasetBuilder
