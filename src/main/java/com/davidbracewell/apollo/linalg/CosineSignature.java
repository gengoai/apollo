package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.affinity.Similarity;

/**
 * The type Cosine signature.
 *
 * @author David B. Bracewell
 */
public class CosineSignature implements SignatureFunction {
  private static final long serialVersionUID = 1L;

  private final int dimension;
  private final int signatureSize;
  private final Vector[] randomProjections;

  /**
   * Instantiates a new Cosine signature.
   *
   * @param signatureSize the signature size
   * @param dimension     the dimension
   */
  public CosineSignature(int signatureSize, int dimension) {
    this.signatureSize = signatureSize;
    this.dimension = dimension;
    this.randomProjections = new Vector[signatureSize];
    for (int i = 0; i < signatureSize; i++) {
      this.randomProjections[i] = SparseVector.randomGaussian(dimension);
    }
  }

  @Override
  public int[] signature(Vector vector) {
    int[] sig = new int[randomProjections.length];
    for (int i = 0; i < signatureSize; i++) {
      sig[i] = randomProjections[i].dot(vector) > 0 ? 1 : 0;
    }
    return sig;
  }

  @Override
  public boolean isBinary() {
    return true;
  }

  @Override
  public int getDimension() {
    return dimension;
  }

  @Override
  public int getSignatureSize() {
    return signatureSize;
  }

  @Override
  public Measure getMeasure() {
    return Similarity.Cosine;
  }

}// END OF CosineSignature
