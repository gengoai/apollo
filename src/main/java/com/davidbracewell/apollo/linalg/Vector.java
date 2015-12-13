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

import com.davidbracewell.Copyable;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.EnhancedDoubleStatistics;
import lombok.NonNull;
import lombok.Value;

import java.io.Serializable;
import java.util.Iterator;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public interface Vector extends Iterable<Vector.Entry>, Copyable<Vector> {
  /**
   * Computes the sum of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be added.
   * @return A new vector whose elements are the sum of this instance and rhs.
   */
  Vector add(Vector rhs);

  /**
   * Computes the sum of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be added.
   * @return This vector.
   */
  Vector addSelf(Vector rhs);

  /**
   * Compresses memory if possible
   */
  void compress();

  /**
   * Decrements the value at the given index.
   *
   * @param index the index to decrement
   * @return This vector
   */
  Vector decrement(int index);

  /**
   * Decrements the value at the given index.
   *
   * @param index  the index to decrement
   * @param amount the amount to decrement by
   * @return This vector
   */
  Vector decrement(int index, double amount);

  /**
   * Returns the dimension of the vector, i.e. the number of elements.
   *
   * @return the dimension of the vector.
   */
  int dimension();

  /**
   * Computes the quotient of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be divided.
   * @return A new vector whose elements are the quotient of this instance and rhs.
   */
  Vector divide(Vector rhs);

  /**
   * Computes the quotient of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be divided.
   * @return This vector.
   */
  Vector divideSelf(Vector rhs);

  /**
   * Compute the dot product of this vector with rhs.
   *
   * @param rhs the vector with which the dot product should be computed.
   * @return the scalar dot product of this instance and rhs.
   */
  double dot(Vector rhs);

  /**
   * Gets the value at the given index.
   *
   * @param index the index of the value wanted
   * @return the value at the given index
   */
  double get(int index);

  /**
   * Increments the value at the given index.
   *
   * @param index the index to increment
   * @return This vector
   */
  Vector increment(int index);

  /**
   * Increments the value at the given index.
   *
   * @param index  the index to increment
   * @param amount the amount to increment by
   * @return This vector
   */
  Vector increment(int index, double amount);

  /**
   * Returns true if this implementation is dense.
   *
   * @return true if this implementation is dense.
   */
  boolean isDense();

  /**
   * Returns true if all elements in the vector are finite.
   *
   * @return true if all elements in the vector are finite.
   */
  boolean isFinite();

  /**
   * Returns true if any element in the vector is <code>Infinite</code>
   *
   * @return true if any element in the vector is <code>Infinite</code>
   */
  boolean isInfinite();

  /**
   * Returns true if any element in the vector is <code>NaN</code>
   *
   * @return true if any element in the vector is <code>NaN</code>
   */
  boolean isNaN();

  /**
   * Returns true if this implementation is sparse.
   *
   * @return true if this implementation is sparse.
   */
  boolean isSparse();

  /**
   * Computes the L1 norm of the vector, which is the sum of the absolute values.
   *
   * @return The L1 norm of the vector
   */
  double l1Norm();

  /**
   * Computes the L-Infinity norm of the vector, which is the max of the absolute values.
   *
   * @return The L-Infinity norm of the vector
   */
  double lInfNorm();

  /**
   * Computes the magnitude (L2 norm) of the vector, which is the square root of the sum of squares.
   *
   * @return the magnitude (L2 norm) of the vector
   */
  double magnitude();

  /**
   * Applies a function to each value in this vector returning a new vector.
   *
   * @param function the function to apply to the values of this vector
   * @return A new vector whose values are the result of the function being applied to this instance.
   */
  Vector map(DoubleUnaryOperator function);

  /**
   * Applies the given function on the elements of this vector and the vector v creates a new vector as a by product.
   *
   * @param v        The vector which is applied as a part of the given function
   * @param function The function to apply
   * @return A new vector whose elements are result of the function applied to the values of this instance and v.
   */
  Vector map(Vector v, DoubleBinaryOperator function);

  /**
   * Adds a given amount to each value in this instance creating a new vector as a result.
   *
   * @param amount the amount to add
   * @return A new vector whose values are the sum this instance and the given amount
   */
  Vector mapAdd(double amount);

  /**
   * Adds a given amount to each value in this instance in place.
   *
   * @param amount the amount to add
   * @return This vector
   */
  Vector mapAddSelf(double amount);

  /**
   * Divides a given amount to each value in this instance creating a new vector as a result.
   *
   * @param amount the amount to divide
   * @return A new vector whose values are the quotient this instance and the given amount
   */
  Vector mapDivide(double amount);

  /**
   * Divides a given amount to each value in this instance in place.
   *
   * @param amount the amount to divide
   * @return This vector.
   */
  Vector mapDivideSelf(double amount);

  /**
   * Multiplies a given amount to each value in this instance creating a new vector as a result.
   *
   * @param amount the amount to multiply
   * @return A new vector whose values are the product this instance and the given amount
   */
  Vector mapMultiply(double amount);

  /**
   * Multiplies a given amount to each value in this instance in place.
   *
   * @param amount the amount to multiply
   * @return This vector.
   */
  Vector mapMultiplySelf(double amount);

  /**
   * Applies the given function on the elements of this vector and the vector v.
   *
   * @param v        The vector which is applied as a part of the given function
   * @param function The function to apply
   * @return This vector.
   */
  Vector mapSelf(Vector v, DoubleBinaryOperator function);

  /**
   * Applies a function to each value in this vector in place.
   *
   * @param function the function to apply to the values of this vector
   * @return This vector.
   */
  Vector mapSelf(DoubleUnaryOperator function);

  /**
   * Subtracts a given amount to each value in this instance creating a new vector as a result.
   *
   * @param amount the amount to subtract
   * @return A new vector whose values are the difference this instance and the given amount
   */
  Vector mapSubtract(double amount);

  /**
   * Subtracts a given amount to each value in this instance in place.
   *
   * @param amount the amount to subtract
   * @return This vector
   */
  Vector mapSubtractSelf(double amount);

  /**
   * Calcualtes the maximum value in this vector.
   *
   * @return The maximum value in this vector.
   */
  double max();

  /**
   * Calculates the minimum value in this vector.
   *
   * @return The minimum value in this vector.
   */
  double min();

  /**
   * Computes the product of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be multiplied.
   * @return A new vector whose elements are the product of this instance and rhs.
   */
  Vector multiply(Vector rhs);

  /**
   * Computes the product of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be multiplied.
   * @return This vector.
   */
  Vector multiplySelf(Vector rhs);

  /**
   * Creates an <code>Iterator</code> over non-zero values in the vector. The order is optimized based on the underlying
   * structure.
   *
   * @return An iterator over non-zero values in the vector.
   */
  Iterator<Vector.Entry> nonZeroIterator();

  /**
   * Creates an <code>Iterator</code> over non-zero values in the vector. The order is in ascending order of index.
   *
   * @return An iterator over non-zero values in the vector.
   */
  Iterator<Vector.Entry> orderedNonZeroIterator();

  /**
   * Scales the value at the given index by a given amount.
   *
   * @param index  the index of the value to scale
   * @param amount the amount to scale by
   * @return This vector
   */
  Vector scale(int index, double amount);

  /**
   * Sets the value of the given index.
   *
   * @param index the index of the value to set
   * @param value the value to set
   * @return This vector
   */
  Vector set(int index, double value);

  /**
   * The current number of values stored in the underlying implementation. Note that size may not equal dimension for
   * sparse implementations.
   *
   * @return the number of values stored in the underlying implementation
   */
  int size();

  /**
   * Constructs a new vector whose dimension is <code>to-from</code> and whose values are come from this vector at
   * indexes <code>from</code> to <code>to</code>. Note that to is not inclusive.
   *
   * @param from Starting point for the slice (inclusive)
   * @param to   Ending point for the slice (not inclusive)
   * @return A new vector whose vales correspond to those in this vector in the indices <code>from</code> to
   * <code>to</code>
   */
  Vector slice(int from, int to);

  /**
   * Calculates simples statistics over the values in the vector.
   *
   * @return simples statistics over the values in the vector.
   */
  EnhancedDoubleStatistics statistics();

  /**
   * Computes the difference of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be subtracted.
   * @return A new vector whose elements are the difference of this instance and rhs.
   */
  Vector subtract(Vector rhs);

  /**
   * Computes the difference of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be subtracted.
   * @return This vector.
   */
  Vector subtractSelf(Vector rhs);

  /**
   * Calculates the sum of the values in this vector.
   *
   * @return The sum of the values in this vector.
   */
  double sum();

  /**
   * Converts the values in the vector to an array.
   *
   * @return An array of the values in the vector.
   */
  double[] toArray();

  /**
   * Sets all elements in the vector to zero.
   *
   * @return This vector
   */
  Vector zero();

  /**
   * Resizes the current vector constructing a new vector
   *
   * @param newDimension the new dimension
   * @return The new vector of the given dimension
   */
  Vector redim(int newDimension);

  default void forEachSparse(@NonNull Consumer<Vector.Entry> consumer) {
    Collect.from(nonZeroIterator()).forEach(consumer);
  }

  default void forEachOrderedSparse(@NonNull Consumer<Vector.Entry> consumer) {
    Collect.from(orderedNonZeroIterator()).forEach(consumer);
  }


  @Value
  class Entry implements Serializable {
    private static final long serialVersionUID = 1L;
    public final int index;
    public final double value;
  }


}//END OF Vector
