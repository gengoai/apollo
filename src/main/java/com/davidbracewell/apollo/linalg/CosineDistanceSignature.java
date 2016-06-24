package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.affinity.Similarity;

/**
 * @author David B. Bracewell
 */
public class CosineDistanceSignature extends CosineSignature {
  private static final long serialVersionUID = 1L;
  /**
   * Instantiates a new Cosine signature.
   *
   * @param signatureSize the signature size
   * @param dimension     the dimension
   */
  public CosineDistanceSignature(int signatureSize, int dimension) {
    super(signatureSize, dimension);
  }

  @Override
  public Measure getMeasure() {
    return Similarity.Cosine.asDistanceMeasure();
  }

}// END OF CosineDistanceSignature
