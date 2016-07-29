package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.Lazy;
import com.davidbracewell.apollo.linalg.Vector;
import org.apache.commons.math3.ml.clustering.Clusterable;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class ApacheClusterable implements Clusterable, Serializable {
  private static final long serialVersionUID = 1L;
  private volatile Vector vector;
  private Lazy<double[]> point = new Lazy<>(() -> vector.toArray());

  public ApacheClusterable(Vector vector) {
    this.vector = vector;
  }

  public Vector getVector() {
    return vector;
  }

  @Override
  public double[] getPoint() {
    return point.get();
  }

  @Override
  public String toString() {
    return vector.toString();
  }

}// END OF ApacheClusterable
