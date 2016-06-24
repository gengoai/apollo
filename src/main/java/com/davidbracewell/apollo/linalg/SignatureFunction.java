package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.linalg.Vector;

import java.io.Serializable;

/**
 * The interface Signature function.
 *
 * @author David B. Bracewell
 */
public interface SignatureFunction extends Serializable {

  /**
   * Signature int [ ].
   *
   * @param vector the vector
   * @return the int [ ]
   */
  int[] signature(Vector vector);

  /**
   * Is binary boolean.
   *
   * @return the boolean
   */
  boolean isBinary();

  /**
   * Gets dimension.
   *
   * @return the dimension
   */
  int getDimension();

  /**
   * Gets signature size.
   *
   * @return the signature size
   */
  int getSignatureSize();

  /**
   * Gets measure.
   *
   * @return the measure
   */
  Measure getMeasure();


}//END OF HashingFunction
