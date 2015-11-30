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

import com.davidbracewell.conversion.Cast;
import com.google.common.base.Preconditions;

import java.util.Arrays;
import java.util.Iterator;

/**
 * The type Dense vector.
 *
 * @author David B. Bracewell
 */
public class DenseVector extends AbstractVector {

  private static final long serialVersionUID = 1L;
  private double[] data;

  /**
   * Instantiates a new Dense vector.
   *
   * @param dimension the dimension of the new vector
   */
  public DenseVector(int dimension) {
    Preconditions.checkArgument(dimension >= 0, "Dimension must be non-negative.");
    this.data = new double[dimension];
  }

  private DenseVector() {

  }

  /**
   * Instantiates a new Dense vector.
   *
   * @param values the values of the vector
   */
  public DenseVector(double[] values) {
    Preconditions.checkNotNull(values);
    this.data = new double[values.length];
    System.arraycopy(values, 0, this.data, 0, values.length);
  }


  /**
   * Copy Constructor
   *
   * @param vector The vector to copy from
   */
  public DenseVector(Vector vector) {
    Preconditions.checkNotNull(vector);
    this.data = vector.toArray();
  }

  /**
   * Static method to create a new <code>DenseVector</code> whose values are 1.
   *
   * @param dimension The dimension of the vector
   * @return a new <code>DenseVector</code> whose values are 1.
   */
  public static Vector ones(int dimension) {
    DenseVector vector = new DenseVector(dimension);
    Arrays.fill(vector.data, 1.0);
    return vector;
  }


  /**
   * Static method to create a new <code>DenseVector</code> whose values wrap an array. Changes to the array will be
   * seen in the vector and vice versa.
   *
   * @param array The array to warp
   * @return a new <code>DenseVector</code> that wraps a given array.
   */
  public static Vector wrap(double[] array) {
    Preconditions.checkNotNull(array);
    DenseVector v = new DenseVector();
    v.data = array;
    return v;
  }

  /**
   * Static method to create a new <code>DenseVector</code> whose values are 0.
   *
   * @param dimension The dimension of the vector
   * @return a new <code>DenseVector</code> whose values are 0.
   */
  public static Vector zeros(int dimension) {
    return new DenseVector(dimension);
  }

  @Override
  public Vector copy() {
    return new DenseVector(this);
  }

  @Override
  public int dimension() {
    return data.length;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (obj instanceof DenseVector) {
      return Arrays.equals(data, Cast.<DenseVector>as(obj).data);
    }
    if (obj instanceof Vector) {
      return Arrays.equals(data, Cast.<Vector>as(obj).toArray());
    }
    return false;
  }

  @Override
  public double get(int index) {
    Preconditions.checkPositionIndex(index, data.length);
    return data[index];
  }

  @Override
  public int hashCode() {
    return 31 * super.hashCode() + Arrays.hashCode(data);
  }

  @Override
  public boolean isDense() {
    return true;
  }

  @Override
  public Vector set(int index, double value) {
    Preconditions.checkPositionIndex(index, data.length);
    data[index] = value;
    return this;
  }

  @Override
  public int size() {
    return data.length;
  }

  @Override
  public Vector slice(int from, int to) {
    Preconditions.checkPositionIndex(from, data.length);
    Preconditions.checkPositionIndex(to, data.length + 1);
    Preconditions.checkState(to > from, "To index must be > from index");
    DenseVector v = new DenseVector((to - from));
    System.arraycopy(this.data, from, v.data, 0, v.dimension());
    return v;
  }

  @Override
  public double[] toArray() {
    double[] array = new double[dimension()];
    System.arraycopy(this.data, 0, array, 0, this.data.length);
    return array;
  }

  @Override
  public String toString() {
    return Arrays.toString(this.data);
  }

  @Override
  public Vector zero() {
    this.data = new double[dimension()];
    return this;
  }

  @Override
  public Vector redim(int newDimension) {
    Vector v = new DenseVector(newDimension);
    for (Iterator<Vector.Entry> itr = nonZeroIterator(); itr.hasNext(); ) {
      Vector.Entry de = itr.next();
      v.set(de.index, de.value);
    }
    return v;
  }

}//END OF DenseVector
