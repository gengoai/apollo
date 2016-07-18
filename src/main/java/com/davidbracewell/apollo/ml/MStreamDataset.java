package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;

import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class MStreamDataset<T extends Example> extends Dataset<T> {
  private final MStream<T> stream;

  /**
   * Instantiates a new Dataset.
   *
   * @param featureEncoder the feature encoder
   * @param labelEncoder   the label encoder
   * @param preprocessors  the preprocessors
   */
  public MStreamDataset(Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors, MStream<T> stream) {
    super(featureEncoder, labelEncoder, preprocessors);
    this.stream = stream;
  }

  @Override
  public DatasetType getType() {
    return DatasetType.Stream;
  }

  @Override
  public StreamingContext getStreamingContext() {
    return stream.getContext();
  }

  @Override
  protected Dataset<T> create(MStream<T> instances, Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors) {
    return new MStreamDataset<>(featureEncoder, labelEncoder, preprocessors, instances);
  }

  @Override
  protected void addAll(MStream<T> stream) {

  }

  @Override
  public MStream<T> stream() {
    return stream;
  }

  @Override
  public Dataset<T> shuffle(Random random) {
    return create(
      stream.shuffle(random),
      getFeatureEncoder(),
      getLabelEncoder(),
      getPreprocessors()
    );
  }

  @Override
  public int size() {
    return (int) stream.count();
  }

}// END OF MStreamDataset
