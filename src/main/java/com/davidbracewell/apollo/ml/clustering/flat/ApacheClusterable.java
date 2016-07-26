package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.Lazy;
import com.davidbracewell.apollo.linalg.LabeledVector;
import org.apache.commons.math3.ml.clustering.Clusterable;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class ApacheClusterable implements Clusterable, Serializable {
  private static final long serialVersionUID = 1L;
  private volatile LabeledVector vector;
  private Lazy<double[]> point = new Lazy<>(() -> vector.toArray());

  public ApacheClusterable(LabeledVector vector) {
    this.vector = vector;
  }

  public LabeledVector getVector() {
    return vector;
  }

  @Override
  public double[] getPoint() {
    return new double[0];
  }

  @Override
  public String toString() {
    return vector.toString();
  }

}// END OF ApacheClusterable
