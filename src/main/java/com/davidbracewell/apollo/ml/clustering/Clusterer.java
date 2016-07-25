package com.davidbracewell.apollo.ml.clustering;

import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Learner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

/**
 * The type Clusterer.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public abstract class Clusterer<T extends Clustering> extends Learner<Instance, T> {
  private static final long serialVersionUID = 1L;
  private EncoderPair encoderPair;


  @Override
  public T train(@NonNull Dataset<Instance> dataset) {
    return super.train(dataset);
  }

  @Override
  protected T trainImpl(Dataset<Instance> dataset) {
    this.encoderPair = dataset.getEncoderPair();
    return cluster(dataset.stream().map(i -> new LabeledVector(i.getLabel(), i.toVector(dataset.getEncoderPair()))));
  }

  /**
   * Cluster t.
   *
   * @param instances the instances
   * @return the t
   */
  public abstract T cluster(MStream<LabeledVector> instances);

  @Override
  public void reset() {
    this.encoderPair = null;
  }


  /**
   * Gets encoder pair.
   *
   * @return the encoder pair
   */
  public EncoderPair getEncoderPair() {
    return encoderPair;
  }

}// END OF Clusterer
