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
import com.davidbracewell.EnhancedDoubleStatistics;
import com.davidbracewell.apollo.affinity.Correlation;
import com.davidbracewell.collection.Streams;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.AtomicDouble;
import lombok.NonNull;
import lombok.Value;

import java.io.Serializable;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * <p>Interface for vectors </p>
 *
 * @author David B. Bracewell
 */
public interface Vector extends Iterable<Vector.Entry>, Copyable<Vector> {

   /**
    * Constructs a new vector which is the element-wise sum of this vector and <code>rhs</code>.
    *
    * @param rhs the vector to be added.
    * @return A new vector whose elements are the sum of this instance and rhs.
    */
   default Vector add(@NonNull Vector rhs) {
      return copy().addSelf(rhs);
   }

   /**
    * Performs ane element-wise addition to the values in this vector using the values of the <code>rhs</code> vector.
    *
    * @param rhs the vector to be added.
    * @return This vector.
    */
   default Vector addSelf(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      rhs.forEachSparse(e -> increment(e.index, e.value));
      return this;
   }

   /**
    * Gets the label associated with the vector if one.
    *
    * @param <T> the label type
    * @return the label or null if none
    */
   default <T> T getLabel() {
      return null;
   }

   /**
    * Convenience method to create a new labeled vector from this vector with the given label.
    *
    * @param label the label to assign to the vector
    * @return the labeled vector
    */
   default Vector withLabel(Object label) {
      return new LabeledVector(label, this);
   }

   /**
    * Constructs a new <code>1 x dimension</code> matrix containing this vector.
    *
    * @return the matrix
    */
   default Matrix toMatrix() {
      return new SparseMatrix(this);
   }

   /**
    * Transpose the vector into a column of a matrix
    *
    * @return the matrix
    */
   default Matrix transpose() {
      return toMatrix().transpose();
   }

   /**
    * Compresses memory if possible
    *
    * @return the vector
    */
   Vector compress();

   /**
    * Decrements the value at the given index.
    *
    * @param index the index to decrement
    * @return This vector
    */
   default Vector decrement(int index) {
      return increment(index, -1);
   }

   /**
    * Decrements the value at the given index.
    *
    * @param index  the index to decrement
    * @param amount the amount to decrement by
    * @return This vector
    */
   default Vector decrement(int index, double amount) {
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
   default Vector divide(@NonNull Vector rhs) {
      return copy().divideSelf(rhs);
   }

   /**
    * Computes the quotient of this vector and rhs in an element-by-element fashion.
    *
    * @param rhs the vector to be divided.
    * @return This vector.
    */
   default Vector divideSelf(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      forEachSparse(e -> scale(e.index, 1d / rhs.get(e.index)));
      return this;
   }

   /**
    * Compute the dot product of this vector with rhs.
    *
    * @param rhs the vector with which the dot product should be computed.
    * @return the scalar dot product of this instance and rhs.
    */
   default double dot(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      AtomicDouble dot = new AtomicDouble(0d);
      Vector small = size() < rhs.size() ? this : rhs;
      Vector big = size() < rhs.size() ? rhs : this;
      small.forEachSparse(e -> dot.addAndGet(e.value * big.get(e.index)));
      return dot.get();
   }

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
   default Vector increment(int index) {
      return increment(index, 1);
   }

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
   default boolean isFinite() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).allMatch(Double::isFinite);
   }

   /**
    * Returns true if any element in the vector is <code>Infinite</code>
    *
    * @return true if any element in the vector is <code>Infinite</code>
    */
   default boolean isInfinite() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).anyMatch(Double::isInfinite);
   }

   /**
    * Returns true if any element in the vector is <code>NaN</code>
    *
    * @return true if any element in the vector is <code>NaN</code>
    */
   default boolean isNaN() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).anyMatch(Double::isNaN);
   }

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
   default double l1Norm() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).map(Math::abs).sum();
   }

   /**
    * Computes the L-Infinity norm of the vector, which is the max of the absolute values.
    *
    * @return The L-Infinity norm of the vector
    */
   default double lInfNorm() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).map(Math::abs).max().orElse(0d);
   }


   /**
    * Computes the magnitude (L2 norm) of the vector, which is the square root of the sum of squares.
    *
    * @return the magnitude (L2 norm) of the vector
    */
   default double magnitude() {
      return Math.sqrt(Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).map(d -> d * d).sum());
   }

   /**
    * Applies a function to each value in this vector returning a new vector.
    *
    * @param function the function to apply to the values of this vector
    * @return A new vector whose values are the result of the function being applied to this instance.
    */
   default Vector map(@NonNull DoubleUnaryOperator function) {
      return copy().mapSelf(function);
   }

   /**
    * Applies the given function on the elements of this vector and the vector v creates a new vector as a by product.
    *
    * @param v        The vector which is applied as a part of the given function
    * @param function The function to apply
    * @return A new vector whose elements are result of the function applied to the values of this instance and v.
    */
   default Vector map(@NonNull Vector v, @NonNull DoubleBinaryOperator function) {
      return copy().mapSelf(v, function);
   }

   /**
    * Adds a given amount to each value in this instance creating a new vector as a result.
    *
    * @param amount the amount to add
    * @return A new vector whose values are the sum this instance and the given amount
    */
   default Vector mapAdd(double amount) {
      return copy().mapAddSelf(amount);
   }

   /**
    * Adds a given amount to each value in this instance in place.
    *
    * @param amount the amount to add
    * @return This vector
    */
   default Vector mapAddSelf(double amount) {
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
   default Vector mapDivide(double amount) {
      return copy().mapDivideSelf(amount);
   }

   /**
    * Divides a given amount to each value in this instance in place.
    *
    * @param amount the amount to divide
    * @return This vector.
    */
   default Vector mapDivideSelf(double amount) {
      forEachSparse(e -> scale(e.index, 1d / amount));
      return this;
   }

   /**
    * Multiplies a given amount to each value in this instance creating a new vector as a result.
    *
    * @param amount the amount to multiply
    * @return A new vector whose values are the product this instance and the given amount
    */
   default Vector mapMultiply(double amount) {
      return copy().mapMultiplySelf(amount);
   }

   /**
    * Multiplies a given amount to each value in this instance in place.
    *
    * @param amount the amount to multiply
    * @return This vector.
    */
   default Vector mapMultiplySelf(double amount) {
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
   default Vector mapSelf(@NonNull Vector v, @NonNull DoubleBinaryOperator function) {
      Preconditions.checkArgument(v.dimension() == dimension(), "Dimension mismatch");
      for (int i = 0; i < dimension(); i++) {
         set(i, function.applyAsDouble(get(i), v.get(i)));
      }
      return this;
   }

   /**
    * Applies a function to each value in this vector in place.
    *
    * @param function the function to apply to the values of this vector
    * @return This vector.
    */
   default Vector mapSelf(@NonNull DoubleUnaryOperator function) {
      for (int i = 0; i < dimension(); i++) {
         set(i, function.applyAsDouble(get(i)));
      }
      return this;
   }

   /**
    * Subtracts a given amount to each value in this instance creating a new vector as a result.
    *
    * @param amount the amount to subtract
    * @return A new vector whose values are the difference this instance and the given amount
    */
   default Vector mapSubtract(double amount) {
      return copy().mapSubtractSelf(amount);
   }

   /**
    * Subtracts a given amount to each value in this instance in place.
    *
    * @param amount the amount to subtract
    * @return This vector
    */
   default Vector mapSubtractSelf(double amount) {
      for (int i = 0; i < dimension(); i++) {
         decrement(i, amount);
      }
      return this;
   }

   /**
    * Calculates the maximum value in this vector.
    *
    * @return The maximum value in this vector.
    */
   default double max() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).max().orElse(0d);
   }

   /**
    * Calculates the minimum value in this vector.
    *
    * @return The minimum value in this vector.
    */
   default double min() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).min().orElse(0d);
   }

   /**
    * Computes the product of this vector and rhs in an element-by-element fashion.
    *
    * @param rhs the vector to be multiplied.
    * @return A new vector whose elements are the product of this instance and rhs.
    */
   default Vector multiply(@NonNull Vector rhs) {
      return copy().multiplySelf(rhs);
   }

   /**
    * Computes the product of this vector and rhs in an element-by-element fashion.
    *
    * @param rhs the vector to be multiplied.
    * @return This vector.
    */
   default Vector multiplySelf(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      forEachSparse(e -> scale(e.index, rhs.get(e.index)));
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
            return new Vector.Entry(index, get(index));
         }
      };
   }

   /**
    * Creates an <code>Iterator</code> over non-zero values in the vector. The order is optimized based on the
    * underlying structure.
    *
    * @return An iterator over non-zero values in the vector.
    */
   default Iterator<Vector.Entry> nonZeroIterator() {
      return orderedNonZeroIterator();
   }

   /**
    * Creates an <code>Iterator</code> over non-zero values in the vector. The order is in ascending order of index.
    *
    * @return An iterator over non-zero values in the vector.
    */
   default Iterator<Vector.Entry> orderedNonZeroIterator() {
      return new Iterator<Entry>() {
         private final PrimitiveIterator.OfInt indexIter = IntStream.range(0, dimension()).iterator();
         private Integer ni = null;

         private boolean advance() {
            while (ni == null) {
               if (indexIter.hasNext()) {
                  ni = indexIter.next();
                  if (get(ni) == 0) {
                     ni = null;
                  }
               } else {
                  return false;
               }
            }
            return ni != null;
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
            return new Vector.Entry(index, get(index));
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
   default Vector scale(int index, double amount) {
      Preconditions.checkPositionIndex(index, dimension());
      set(index, get(index) * amount);
      return this;
   }

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
   default EnhancedDoubleStatistics statistics() {
      return DoubleStream.of(toArray()).collect(EnhancedDoubleStatistics::new, EnhancedDoubleStatistics::accept,
                                                EnhancedDoubleStatistics::combine);
   }

   /**
    * Computes the difference of this vector and rhs in an element-by-element fashion.
    *
    * @param rhs the vector to be subtracted.
    * @return A new vector whose elements are the difference of this instance and rhs.
    */
   default Vector subtract(@NonNull Vector rhs) {
      return copy().subtractSelf(rhs);
   }

   /**
    * Computes the difference of this vector and rhs in an element-by-element fashion.
    *
    * @param rhs the vector to be subtracted.
    * @return This vector.
    */
   default Vector subtractSelf(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      rhs.forEachSparse(e -> decrement(e.index, e.value));
      return this;
   }

   /**
    * Calculates the sum of the values in this vector.
    *
    * @return The sum of the values in this vector.
    */
   default double sum() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).sum();
   }

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

   /**
    * Applies the given consumer to each non-zero element in this vector
    *
    * @param consumer the consumer to run
    */
   default void forEachSparse(@NonNull Consumer<Vector.Entry> consumer) {
      Streams.asStream(nonZeroIterator()).forEach(consumer);
   }

   /**
    * Applies the given consumer to each non-zero element in this vector with granted order.
    *
    * @param consumer the consumer to run
    */
   default void forEachOrderedSparse(@NonNull Consumer<Vector.Entry> consumer) {
      Streams.asStream(orderedNonZeroIterator()).forEach(consumer);
   }


   /**
    * Determines the Spearman correlation between this vector and the given other vector
    *
    * @param other the other vector to calculate the correlation lwith
    * @return the Spearman correlation
    */
   default double corr(@NonNull Vector other) {
      return Correlation.Spearman.calculate(this, other);
   }

   /**
    * Defines an entry in the vector using its coordinate (index) and its value
    */
   @Value
   class Entry implements Serializable {
      private static final long serialVersionUID = 1L;
      /**
       * The Index.
       */
      public final int index;
      /**
       * The Value.
       */
      public final double value;
   }

}//END OF Vector
