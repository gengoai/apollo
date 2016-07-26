package com.davidbracewell.apollo.ml.clustering.flat;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class ApacheDistanceMeasure implements DistanceMeasure, Serializable {
  private static final long serialVersionUID = 1L;
  private final com.davidbracewell.apollo.affinity.DistanceMeasure wrapped;

  public ApacheDistanceMeasure(com.davidbracewell.apollo.affinity.DistanceMeasure wrapped) {
    this.wrapped = wrapped;
  }

  @Override
  public double compute(double[] doubles, double[] doubles1) throws DimensionMismatchException {
    return wrapped.calculate(doubles, doubles1);
  }

  public com.davidbracewell.apollo.affinity.DistanceMeasure getWrapped() {
    return wrapped;
  }
}// END OF ApacheDistanceMeasure
