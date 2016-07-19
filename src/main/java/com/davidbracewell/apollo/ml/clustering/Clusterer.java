package com.davidbracewell.apollo.ml.clustering;

import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Learner;
import lombok.NonNull;

import java.util.LinkedList;
import java.util.List;

/**
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
    List<LabeledVector> instances = new LinkedList<>();
    dataset.forEach(i -> instances.add(new LabeledVector(i.getLabel(), i.toVector(dataset.getEncoderPair()))));
    return cluster(instances);
  }

  public abstract T cluster(List<LabeledVector> instances);

  @Override
  public void reset() {
    this.encoderPair = null;
  }


  public EncoderPair getEncoderPair() {
    return encoderPair;
  }

}// END OF Clusterer
