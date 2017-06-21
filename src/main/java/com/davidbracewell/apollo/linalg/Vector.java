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
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;
import lombok.Value;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * <p>Interface for vectors </p>
 *
 * @author David B. Bracewell
 */
public interface Vector extends Iterable<Vector.Entry>, Copyable<Vector> {


   /**
    * D ones vector.
    *
    * @param dimension the dimension
    * @return the vector
    */
   static Vector dOnes(int dimension) {
      return DenseVector.ones(dimension);
   }

   /**
    * D random vector.
    *
    * @param dimension the dimension
    * @param min       the min
    * @param max       the max
    * @return the vector
    */
   static Vector dRandom(int dimension, int min, int max) {
      return DenseVector.random(dimension, min, max);
   }

   /**
    * D zeros vector.
    *
    * @param dimension the dimension
    * @return the vector
    */
   static Vector dZeros(int dimension) {
      return DenseVector.zeros(dimension);
   }

   /**
    * S ones vector.
    *
    * @param dimension the dimension
    * @return the vector
    */
   static Vector sOnes(int dimension) {
      return SparseVector.ones(dimension);
   }

   /**
    * S random vector.
    *
    * @param dimension the dimension
    * @param min       the min
    * @param max       the max
    * @return the vector
    */
   static Vector sRandom(int dimension, int min, int max) {
      return SparseVector.random(dimension, min, max);
   }

   /**
    * S zeros vector.
    *
    * @param dimension the dimension
    * @return the vector
    */
   static Vector sZeros(int dimension) {
      return SparseVector.zeros(dimension);
   }

   /**
    * Constructs a new vector which is the element-wise sum of this vector and <code>rhs</code>.
    *
    * @param rhs the vector to be added.
    * @return A new vector whose elements are the sum of this instance and rhs.
    */
   Vector add(@NonNull Vector rhs);

   /**
    * Performs ane element-wise addition to the values in this vector using the values of the <code>rhs</code> vector.
    *
    * @param rhs the vector to be added.
    * @return This vector.
    */
   Vector addSelf(@NonNull Vector rhs);

   /**
    * As map map.
    *
    * @return the map
    */
   Map<Integer, Double> asMap();

   /**
    * Compresses memory if possible
    *
    * @return the vector
    */
   Vector compress();

   /**
    * Determines the Spearman correlation between this vector and the given other vector
    *
    * @param other the other vector to calculate the correlation lwith
    * @return the Spearman correlation
    */
   double corr(@NonNull Vector other);

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
    * Returns the k of the vector, i.e. the number of elements.
    *
    * @return the k of the vector.
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
   Vector divideSelf(@NonNull Vector rhs);

   /**
    * Compute the dot product of this vector with rhs.
    *
    * @param rhs the vector with which the dot product should be computed.
    * @return the scalar dot product of this instance and rhs.
    */
   double dot(@NonNull Vector rhs);

   /**
    * Applies the given consumer to each non-zero element in this vector with granted order.
    *
    * @param consumer the consumer to run
    */
   void forEachOrderedSparse(@NonNull Consumer<Vector.Entry> consumer);

   /**
    * Applies the given consumer to each non-zero element in this vector
    *
    * @param consumer the consumer to run
    */
   void forEachSparse(@NonNull Consumer<Vector.Entry> consumer);

   /**
    * Gets the value at the given index.
    *
    * @param index the index of the value wanted
    * @return the value at the given index
    */
   double get(int index);

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
    * Gets label vector.
    *
    * @param dimension the dimension
    * @return the label vector
    */
   default Vector getLabelVector(int dimension) {
      Object label = getLabel();
      if (label == null) {
         return null;
      } else if (label instanceof Vector) {
         Vector v = Cast.as(label);
         if (v.dimension() == dimension) {
            return v;
         }
         return v.redim(dimension);
      } else if (label instanceof Number) {
         if (dimension == 1) {
            return new SinglePointVector(Cast.<Number>as(label).doubleValue());
         }
         return SparseVector.zeros(dimension).set(Cast.<Number>as(label).intValue(), 1.0);
      }
      throw new IllegalStateException(label.getClass() + " cannot be converted to a vector");
   }

   /**
    * Gets label vector.
    *
    * @return the label vector
    */
   default Vector getLabelVector() {
      Object label = getLabel();
      if (label == null) {
         return null;
      } else if (label instanceof Vector) {
         return Cast.as(label);
      } else if (label instanceof Number) {
         return new SinglePointVector(Cast.<Number>as(label).doubleValue());
      }
      throw new IllegalStateException(label.getClass() + " cannot be converted to a vector");
   }

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
    * Insert vector.
    *
    * @param i the
    * @param v the v
    * @return the vector
    */
   Vector insert(int i, double v);

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
   Vector map(@NonNull DoubleUnaryOperator function);

   /**
    * Applies the given function on the elements of this vector and the vector v creates a new vector as a by product.
    *
    * @param v        The vector which is applied as a part of the given function
    * @param function The function to apply
    * @return A new vector whose elements are result of the function applied to the values of this instance and v.
    */
   Vector map(@NonNull Vector v, @NonNull DoubleBinaryOperator function);

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
   Vector mapSelf(@NonNull Vector v, @NonNull DoubleBinaryOperator function);

   /**
    * Applies a function to each value in this vector in place.
    *
    * @param function the function to apply to the values of this vector
    * @return This vector.
    */
   Vector mapSelf(@NonNull DoubleUnaryOperator function);

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
    * Calculates the maximum value in this vector.
    *
    * @return The maximum value in this vector.
    */
   double max();

   /**
    * Max index int.
    *
    * @return the int
    */
   int maxIndex();

   /**
    * Calculates the minimum value in this vector.
    *
    * @return The minimum value in this vector.
    */
   double min();

   /**
    * Min index int.
    *
    * @return the int
    */
   int minIndex();

   /**
    * Computes the product of this vector and rhs in an element-by-element fashion.
    *
    * @param rhs the vector to be multiplied.
    * @return A new vector whose elements are the product of this instance and rhs.
    */
   Vector multiply(@NonNull Vector rhs);

   /**
    * Computes the product of this vector and rhs in an element-by-element fashion.
    *
    * @param rhs the vector to be multiplied.
    * @return This vector.
    */
   Vector multiplySelf(@NonNull Vector rhs);

   /**
    * Creates an <code>Iterator</code> over non-zero values in the vector. The order is optimized based on the
    * underlying structure.
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
    * Resizes the current vector constructing a new vector
    *
    * @param newDimension the new k
    * @return The new vector of the given k
    */
   Vector redim(int newDimension);

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
    * The current number of values stored in the underlying implementation. Note that size may not equal k for
    * sparse implementations.
    *
    * @return the number of values stored in the underlying implementation
    */
   int size();

   /**
    * Constructs a new vector whose k is <code>to-from</code> and whose values are come from this vector at
    * indexes <code>from</code> to <code>to</code>. Note that to is not inclusive.
    *
    * @param from Starting point for the slice (inclusive)
    * @param to   Ending point for the slice (not inclusive)
    * @return A new vector whose vales correspond to those in this vector in the indices <code>from</code> to
    * <code>to</code>
    */
   Vector slice(int from, int to);

   /**
    * Sparse tuple 2.
    *
    * @return the tuple 2
    */
   Tuple2<int[], double[]> sparse();

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
   Vector subtract(@NonNull Vector rhs);

   /**
    * Computes the difference of this vector and rhs in an element-by-element fashion.
    *
    * @param rhs the vector to be subtracted.
    * @return This vector.
    */
   Vector subtractSelf(@NonNull Vector rhs);

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
    * Constructs a new <code>k x k</code> matrix with the elements of this vector on the diagonal.
    *
    * @return the matrix
    */
   Matrix toDiagMatrix();

   /**
    * Constructs a new <code>1 x k</code> matrix containing this vector.
    *
    * @return the matrix
    */
   Matrix toMatrix();

   /**
    * To unit vector vector.
    *
    * @return the vector
    */
   Vector toUnitVector();

   /**
    * Transpose the vector into a column of a matrix
    *
    * @return the matrix
    */
   Matrix transpose();

   /**
    * Convenience method to create a new labeled vector from this vector with the given label.
    *
    * @param label the label to assign to the vector
    * @return the labeled vector
    */
   Vector withLabel(Object label);

   /**
    * Sets all elements in the vector to zero.
    *
    * @return This vector
    */
   Vector zero();

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
