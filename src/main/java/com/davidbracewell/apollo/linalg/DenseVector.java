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

import com.google.common.base.Preconditions;
import lombok.EqualsAndHashCode;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Arrays;

/**
 * The type Dense vector.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = false)
public class DenseVector implements Vector, Serializable {
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
  public DenseVector(@NonNull double[] values) {
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
  public static Vector wrap(@NonNull double[] array) {
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
  public Vector compress() {
    return this;
  }

  @Override
  public int dimension() {
    return this.data.length;
  }

  @Override
  public double get(int index) {
    return this.data[index];
  }

  @Override
  public Vector increment(int index, double amount) {
    Preconditions.checkPositionIndex(index, dimension());
    this.data[index] += amount;
    return this;
  }

  @Override
  public boolean isDense() {
    return true;
  }

  @Override
  public boolean isSparse() {
    return false;
  }

  @Override
  public Vector set(int index, double value) {
    Preconditions.checkPositionIndex(index, dimension());
    this.data[index] = value;
    return this;
  }

  @Override
  public int size() {
    return this.data.length;
  }

  @Override
  public Vector slice(int from, int to) {
    Preconditions.checkPositionIndex(from, dimension());
    Preconditions.checkPositionIndex(to, dimension() + 1);
    Preconditions.checkState(to > from, "To index must be > from index");
    DenseVector copy = new DenseVector(to - from);
    copy.data = Arrays.copyOfRange(this.data, from, to);
    return copy;
  }

  @Override
  public double[] toArray() {
    return this.data;
  }

  @Override
  public Vector zero() {
    this.data = new double[this.data.length];
    return this;
  }

  @Override
  public Vector redim(int newDimension) {
    DenseVector copy = new DenseVector(newDimension);
    copy.data = Arrays.copyOf(this.data, newDimension);
    return copy;
  }

  @Override
  public Vector copy() {
    return new DenseVector(this);
  }


  @Override
  public String toString() {
    return Arrays.toString(toArray());
  }
}//END OF DenseVector
