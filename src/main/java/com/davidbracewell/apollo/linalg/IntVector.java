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
import com.google.common.util.concurrent.AtomicDouble;
import lombok.NonNull;
import lombok.Value;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.function.Consumer;
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/**
 * The interface Int vector.
 *
 * @author David B. Bracewell
 */
public interface IntVector extends Iterable<IntVector.Entry>, Copyable<IntVector> {
  /**
   * Computes the sum of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be added.
   * @return A new vector whose elements are the sum of this instance and rhs.
   */
  default IntVector add(IntVector rhs) {
    IntVector iv = copy();
    return iv.addSelf(rhs);
  }

  /**
   * Computes the sum of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be added.
   * @return This vector.
   */
  default IntVector addSelf(@NonNull IntVector rhs) {
    rhs.forEachSparse(e -> increment(e.index, e.value));
    return this;
  }

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
  default IntVector decrement(int index) {
    return increment(index, -1);
  }

  /**
   * Decrements the value at the given index.
   *
   * @param index  the index to decrement
   * @param amount the amount to decrement by
   * @return This vector
   */
  default IntVector decrement(int index, int amount) {
    return increment(index, -amount);
  }

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
  default IntVector divide(IntVector rhs) {
    IntVector iv = copy();
    return iv.divide(rhs);
  }

  /**
   * Computes the quotient of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be divided.
   * @return This vector.
   */
  default IntVector divideSelf(IntVector rhs) {
    rhs.forEachSparse(e -> set(e.index, get(e.index) / e.getValue()));
    return this;
  }

  /**
   * Compute the dot product of this vector with rhs.
   *
   * @param rhs the vector with which the dot product should be computed.
   * @return the scalar dot product of this instance and rhs.
   */
  default double dot(IntVector rhs) {
    AtomicDouble dot = new AtomicDouble();
    rhs.forEachSparse(e -> dot.addAndGet(e.getValue() * get(e.index)));
    return dot.get();
  }

  /**
   * Gets the value at the given index.
   *
   * @param index the index of the value wanted
   * @return the value at the given index
   */
  int get(int index);

  /**
   * Increments the value at the given index.
   *
   * @param index the index to increment
   * @return This vector
   */
  default IntVector increment(int index) {
    return increment(index, 1);
  }

  /**
   * Increments the value at the given index.
   *
   * @param index  the index to increment
   * @param amount the amount to increment by
   * @return This vector
   */
  IntVector increment(int index, int amount);

  /**
   * Applies a function to each value in this vector returning a new vector.
   *
   * @param function the function to apply to the values of this vector
   * @return A new vector whose values are the result of the function being applied to this instance.
   */
  default IntVector map(IntUnaryOperator function) {
    IntVector iv = copy();
    return iv.mapSelf(function);
  }

  /**
   * Applies the given function on the elements of this vector and the vector v creates a new vector as a by product.
   *
   * @param v        The vector which is applied as a part of the given function
   * @param function The function to apply
   * @return A new vector whose elements are result of the function applied to the values of this instance and v.
   */
  default IntVector map(IntVector v, IntBinaryOperator function) {
    IntVector iv = copy();
    return iv.mapSelf(v, function);
  }

  /**
   * Adds a given amount to each value in this instance creating a new vector as a result.
   *
   * @param amount the amount to add
   * @return A new vector whose values are the sum this instance and the given amount
   */
  default IntVector mapAdd(int amount) {
    IntVector iv = copy();
    return iv.mapAddSelf(amount);
  }

  /**
   * Adds a given amount to each value in this instance in place.
   *
   * @param amount the amount to add
   * @return This vector
   */
  default IntVector mapAddSelf(int amount) {
    for (int i = 0; i < dimension(); i++) {
      increment(i, amount);
    }
    return this;
  }

  /**
   * Divides a given amount to each value in this instance creating a new vector as a result.
   *
   * @param amount the amount to divide
   * @return A new vector whose values are the quotient this instance and the given amount
   */
  default IntVector mapDivide(int amount) {
    IntVector iv = copy();
    return iv.mapDivideSelf(amount);
  }

  /**
   * Divides a given amount to each value in this instance in place.
   *
   * @param amount the amount to divide
   * @return This vector.
   */
  default IntVector mapDivideSelf(int amount) {
    forEachSparse(e -> set(e.index, e.getValue() / amount));
    return this;
  }

  /**
   * Multiplies a given amount to each value in this instance creating a new vector as a result.
   *
   * @param amount the amount to multiply
   * @return A new vector whose values are the product this instance and the given amount
   */
  default IntVector mapMultiply(int amount) {
    IntVector iv = copy();
    return iv.mapMultiplySelf(amount);
  }

  /**
   * Multiplies a given amount to each value in this instance in place.
   *
   * @param amount the amount to multiply
   * @return This vector.
   */
  default IntVector mapMultiplySelf(int amount) {
    forEachSparse(e -> scale(e.index, amount));
    return this;
  }

  /**
   * Applies the given function on the elements of this vector and the vector v.
   *
   * @param v        The vector which is applied as a part of the given function
   * @param function The function to apply
   * @return This vector.
   */
  default IntVector mapSelf(@NonNull IntVector v, @NonNull IntBinaryOperator function) {
    for (int i = 0; i < dimension(); i++) {
      set(i, function.applyAsInt(get(i), v.get(i)));
    }
    return this;
  }

  /**
   * Applies a function to each value in this vector in place.
   *
   * @param function the function to apply to the values of this vector
   * @return This vector.
   */
  default IntVector mapSelf(@NonNull IntUnaryOperator function) {
    for (int i = 0; i < dimension(); i++) {
      set(i, function.applyAsInt(get(i)));
    }
    return this;
  }

  /**
   * Subtracts a given amount to each value in this instance creating a new vector as a result.
   *
   * @param amount the amount to subtract
   * @return A new vector whose values are the difference this instance and the given amount
   */
  default IntVector mapSubtract(int amount) {
    IntVector iv = copy();
    return iv.mapSubtractSelf(amount);
  }

  /**
   * Subtracts a given amount to each value in this instance in place.
   *
   * @param amount the amount to subtract
   * @return This vector
   */
  default IntVector mapSubtractSelf(int amount) {
    for (int i = 0; i < dimension(); i++) {
      decrement(i, amount);
    }
    return this;
  }

  /**
   * Calcualtes the maximum value in this vector.
   *
   * @return The maximum value in this vector.
   */
  default int max() {
    return IntStream.of(toArray()).max().orElse(0);
  }

  /**
   * Calculates the minimum value in this vector.
   *
   * @return The minimum value in this vector.
   */
  default int min() {
    return IntStream.of(toArray()).max().orElse(0);
  }

  /**
   * Computes the product of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be multiplied.
   * @return A new vector whose elements are the product of this instance and rhs.
   */
  default IntVector multiply(IntVector rhs) {
    IntVector iv = copy();
    return iv.multiplySelf(rhs);
  }

  /**
   * Computes the product of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be multiplied.
   * @return This vector.
   */
  default IntVector multiplySelf(@NonNull IntVector rhs) {
    rhs.forEachSparse(e -> scale(e.index, e.value));
    return this;
  }

  @Override
  default Iterator<Entry> iterator() {
    return new Iterator<Entry>() {
      private final PrimitiveIterator.OfInt indexIter = IntStream.range(0, dimension()).iterator();

      @Override
      public boolean hasNext() {
        return indexIter.hasNext();
      }

      @Override
      public Entry next() {
        if (!indexIter.hasNext()) {
          throw new NoSuchElementException();
        }
        int index = indexIter.next();
        return new IntVector.Entry(index, get(index));
      }
    };
  }

  /**
   * Creates an <code>Iterator</code> over non-zero values in the vector. The order is optimized based on the
   * underlying
   * structure.
   *
   * @return An iterator over non-zero values in the vector.
   */
  default Iterator<IntVector.Entry> nonZeroIterator() {
    return orderedNonZeroIterator();
  }

  /**
   * Creates an <code>Iterator</code> over non-zero values in the vector. The order is in ascending order of index.
   *
   * @return An iterator over non-zero values in the vector.
   */
  default Iterator<IntVector.Entry> orderedNonZeroIterator() {
    return new Iterator<Entry>() {
      private final PrimitiveIterator.OfInt indexIter = IntStream.range(0, dimension()).iterator();
      private Integer ni = null;

      private boolean advance() {
        if (ni == null) {
          while (indexIter.hasNext()) {
            int i = indexIter.next();
            if (get(i) != 0) {
              ni = i;
              return true;
            }
          }
        }
        return false;
      }

      @Override
      public boolean hasNext() {
        return advance();
      }

      @Override
      public Entry next() {
        if (!advance()) {
          throw new NoSuchElementException();
        }
        int index = ni;
        ni = null;
        return new IntVector.Entry(index, get(index));
      }
    };
  }

  /**
   * Scales the value at the given index by a given amount.
   *
   * @param index  the index of the value to scale
   * @param amount the amount to scale by
   * @return This vector
   */
  IntVector scale(int index, int amount);

  /**
   * Sets the value of the given index.
   *
   * @param index the index of the value to set
   * @param value the value to set
   * @return This vector
   */
  IntVector set(int index, int value);

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
  IntVector slice(int from, int to);

  /**
   * Calculates simples statistics over the values in the vector.
   *
   * @return simples statistics over the values in the vector.
   */
  default EnhancedDoubleStatistics statistics() {
    return IntStream.of(toArray()).asDoubleStream().collect(EnhancedDoubleStatistics::new, EnhancedDoubleStatistics::accept, EnhancedDoubleStatistics::combine);
  }

  /**
   * Computes the difference of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be subtracted.
   * @return A new vector whose elements are the difference of this instance and rhs.
   */
  default IntVector subtract(IntVector rhs) {
    IntVector iv = copy();
    return iv.subtractSelf(rhs);
  }

  /**
   * Computes the difference of this vector and rhs in an element-by-element fashion.
   *
   * @param rhs the vector to be subtracted.
   * @return This vector.
   */
  default IntVector subtractSelf(@NonNull IntVector rhs) {
    rhs.forEachSparse(e -> decrement(e.index, e.value));
    return this;
  }

  /**
   * Calculates the sum of the values in this vector.
   *
   * @return The sum of the values in this vector.
   */
  default int sum() {
    return IntStream.of(toArray()).sum();
  }

  /**
   * Converts the values in the vector to an array.
   *
   * @return An array of the values in the vector.
   */
  int[] toArray();

  /**
   * Sets all elements in the vector to zero.
   *
   * @return This vector
   */
  IntVector zero();

  /**
   * Resizes the current vector constructing a new vector
   *
   * @param newDimension the new dimension
   * @return The new vector of the given dimension
   */
  IntVector redim(int newDimension);

  /**
   * For each sparse.
   *
   * @param consumer the consumer
   */
  default void forEachSparse(@NonNull Consumer<IntVector.Entry> consumer) {
    Collect.from(nonZeroIterator()).forEach(consumer);
  }

  /**
   * For each ordered sparse.
   *
   * @param consumer the consumer
   */
  default void forEachOrderedSparse(@NonNull Consumer<IntVector.Entry> consumer) {
    Collect.from(orderedNonZeroIterator()).forEach(consumer);
  }

  /**
   * The type Entry.
   */
  @Value
  class Entry {
    /**
     * The Index.
     */
    public int index;
    /**
     * The Value.
     */
    public int value;
  }

}//END OF IntVector
