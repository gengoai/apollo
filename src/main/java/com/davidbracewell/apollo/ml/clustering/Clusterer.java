package com.davidbracewell.apollo.ml.clustering;

import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Learner;

import java.util.LinkedList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public abstract class Clusterer extends Learner<Instance, Clustering> {
  private static final long serialVersionUID = 1L;
  private EncoderPair encoderPair;

  @Override
  protected Clustering trainImpl(Dataset<Instance> dataset) {
    this.encoderPair = dataset.getEncoderPair();
    List<LabeledVector> instances = new LinkedList<>();
    dataset.forEach(i -> instances.add(new LabeledVector(i.getLabel(), i.toVector(dataset.getEncoderPair()))));
    return cluster(instances);
  }

  public abstract Clustering cluster(List<LabeledVector> instances);

  @Override
  public void reset() {
    this.encoderPair = null;
  }


  public EncoderPair getEncoderPair() {
    return encoderPair;
  }

}// END OF Clusterer
