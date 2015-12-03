package com.davidbracewell.apollo.ml;

import com.davidbracewell.stream.MStream;
import lombok.NonNull;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public class DatasetBuilder<T extends Example> {
  @Setter(onParam = @_({@NonNull}))
  private Dataset.Type type = Dataset.Type.InMemory;
  @Setter(onParam = @_({@NonNull}))
  private FeatureEncoder featureEncoder = new IndexFeatureEncoder();
  @Setter(onParam = @_({@NonNull}))
  private LabelEncoder labelEncoder = new DiscreteLabelEncoder();
  @Setter
  private MStream<T> streamSource;


  public Dataset<T> build() {
    Dataset<T> dataset;
    switch (type) {
      case Distributed:
      case OffHeap:
        dataset = new OffHeapDataset<>(featureEncoder, labelEncoder);
        break;
      default:
        dataset = new InMemoryDataset<>(featureEncoder, labelEncoder);
    }

    if (streamSource != null) {
      dataset.addAll(streamSource);
    }

    return dataset;
  }

}// END OF DatasetBuilder
