package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.affinity.Similarity;

/**
 * The type Min hash distance signature.
 *
 * @author David B. Bracewell
 */
public class MinHashDistanceSignature extends MinHashSignature {
  /**
   * Instantiates a new Min hash distance signature.
   *
   * @param error     the error
   * @param dimension the dimension
   */
  public MinHashDistanceSignature(double error, int dimension) {
    super(error, dimension);
  }

  @Override
  public Measure getMeasure() {
    return Similarity.Jaccard.asDistanceMeasure();
  }
}// END OF MinHashDistanceSignature
