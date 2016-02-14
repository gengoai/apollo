/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Convert;
import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.BitSet;
import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class SparseBitVector implements Vector, Serializable {

  private static final long serialVersionUID = 3841985936614861097L;
  private final BitSet bits;
  private final int dimension;

  /**
   * Instantiates a new SparseBitVector.
   *
   * @param dimension the dimension of the new vector
   */
  public SparseBitVector(int dimension) {
    Preconditions.checkArgument(dimension >= 0);
    this.dimension = dimension;
    this.bits = new BitSet(dimension);
  }

  /**
   * Copy Constructor
   *
   * @param vector The vector to copy from
   */
  public SparseBitVector(Vector vector) {
    this.dimension = Preconditions.checkNotNull(vector.dimension());
    this.bits = new BitSet(this.dimension);
    for (Iterator<Vector.Entry> itr = vector.nonZeroIterator(); itr.hasNext(); ) {
      Vector.Entry de = itr.next();
      set(de.index, de.value);
    }
  }

  @Override
  public Vector copy() {
    return new SparseBitVector(this);
  }


  @Override
  public Vector compress() {
    return this;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public double get(int index) {
    return bits.get(index) ? 1.0 : 0d;
  }

  @Override
  public Vector increment(int index, double amount) {
    bits.set(index, (bits.get(index) ? 1.0 : 0.0) + amount > 0);
    return this;
  }

  @Override
  public boolean isDense() {
    return false;
  }

  @Override
  public boolean isSparse() {
    return true;
  }

  @Override
  public double l1Norm() {
    return bits.cardinality();
  }

  @Override
  public double lInfNorm() {
    return max();
  }

  @Override
  public double magnitude() {
    return Math.sqrt(bits.cardinality());
  }

  @Override
  public double max() {
    return bits.cardinality() > 0 ? 1d : 0d;
  }

  @Override
  public double min() {
    return bits.cardinality() == dimension ? 1d : 0d;
  }

  @Override
  public Vector set(int index, double value) {
    if (value > 0) {
      bits.set(index);
    } else {
      bits.clear(index);
    }
    return this;
  }

  @Override
  public int size() {
    return bits.size();
  }

  @Override
  public Vector slice(int from, int to) {
    Preconditions.checkPositionIndex(from, dimension);
    Preconditions.checkPositionIndex(to, dimension + 1);
    Preconditions.checkState(to > from, "To index must be > from index");
    SparseBitVector v = new SparseBitVector((to - from));
    for (int i = from; i < to; i++) {
      v.set(i, get(i));
    }
    return v;
  }

  @Override
  public double sum() {
    return bits.cardinality();
  }

  @Override
  public double[] toArray() {
    return Convert.convert(bits.toLongArray(), double[].class);
  }

  @Override
  public Vector zero() {
    bits.clear();
    return this;
  }

  @Override
  public Vector redim(int newDimension) {
    Vector v = new SparseBitVector(newDimension);
    for (Iterator<Vector.Entry> itr = nonZeroIterator(); itr.hasNext(); ) {
      Vector.Entry de = itr.next();
      v.set(de.index, de.value);
    }
    return v;
  }

}//END OF SparseBitVector
