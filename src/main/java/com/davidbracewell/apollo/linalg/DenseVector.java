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
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * <p>A Vector implementation backed by a double array</p>
 *
 * @author David B. Bracewell
 */
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
   public DenseVector(@NonNull double... values) {
      this.data = new double[values.length];
      System.arraycopy(values, 0, this.data, 0, values.length);
   }


   /**
    * Copy Constructor
    *
    * @param vector The vector to copy from
    */
   public DenseVector(@NonNull Vector vector) {
      this(vector.toArray());
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
   public static Vector wrap(@NonNull double... array) {
      DenseVector v = new DenseVector();
      v.data = array;
      return v;
   }


   /**
    * Creates a new vector of the given dimension initialized with random values in the range of <code>[min,
    * max]</code>.
    *
    * @param dimension the dimension of the vector
    * @param min       the minimum assignable value
    * @param max       the maximum assignable value
    * @return the vector
    */
   public static Vector random(int dimension, double min, double max) {
      return random(dimension, min, max, new Well19937c());
   }


   /**
    * Creates a new vector of the given dimension initialized with random values in the range of <code>[min,
    * max]</code>.
    *
    * @param dimension the dimension of the vector
    * @param min       the minimum assignable value
    * @param max       the maximum assignable value
    * @param rnd       the random number generator to use generate values
    * @return the vector
    */
   public static Vector random(int dimension, double min, double max, @NonNull RandomGenerator rnd) {
      Preconditions.checkArgument(dimension >= 0, "Dimension must be non-negative");
      Preconditions.checkArgument(max > min, "Invalid Range [" + min + ", " + max + "]");
      DenseVector v = new DenseVector(dimension);
      for (int i = 0; i < dimension; i++) {
         v.set(i, rnd.nextDouble() * (max - min) + min);
      }
      return v;
   }

   /**
    * Constructs a new vector of given dimension with values randomized using a gaussian with mean 0 and standard
    * deviation of 1
    *
    * @param dimension the dimension of the vector
    * @return the vector
    */
   public static Vector randomGaussian(int dimension) {
      DenseVector v = new DenseVector(dimension);
      Random rnd = new Random();
      for (int i = 0; i < dimension; i++) {
         v.set(i, rnd.nextGaussian());
      }
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

   @Override
   public boolean equals(Object o) {
      return o != null && o instanceof Vector && Arrays.equals(toArray(), Cast.<Vector>as(o).toArray());
   }

   @Override
   public int hashCode() {
      return Arrays.hashCode(toArray());
   }

}//END OF DenseVector
