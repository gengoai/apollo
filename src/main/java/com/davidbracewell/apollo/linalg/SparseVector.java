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

import com.davidbracewell.Math2;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntIterator;
import lombok.NonNull;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.function.Consumer;

/**
 * A sparse vector implementation backed by a map
 *
 * @author David B. Bracewell
 */
public class SparseVector implements Vector, Serializable {
   private static final long serialVersionUID = 1L;
   private final Int2DoubleOpenHashMap map;
   private final int dimension;


   public SparseVector(double[] dense) {
      this(DenseVector.wrap(dense));
   }

   /**
    * Instantiates a new Sparse vector.
    *
    * @param dimension the dimension of the new vector
    */
   public SparseVector(int dimension) {
      Preconditions.checkArgument(dimension >= 0, "Dimension must be non-negative.");
      this.dimension = dimension;
      this.map = new Int2DoubleOpenHashMap();
   }

   /**
    * Copy Constructor
    *
    * @param vector The vector to copy from
    */
   public SparseVector(@NonNull Vector vector) {
      this.dimension = vector.dimension();
      this.map = new Int2DoubleOpenHashMap(vector.size());
      for (Iterator<Vector.Entry> itr = vector.nonZeroIterator(); itr.hasNext(); ) {
         Vector.Entry de = itr.next();
         this.map.put(de.index, de.value);
      }
   }

   /**
    * Static method to create a new <code>SparseVector</code> whose values are 1.
    *
    * @param dimension The dimension of the vector
    * @return a new <code>SparseVector</code> whose values are 1.
    */
   public static Vector ones(int dimension) {
      Vector v = new SparseVector(dimension);
      for (int i = 0; i < dimension; i++) {
         v.set(i, 1);
      }
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
      SparseVector v = new SparseVector(dimension);
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
      SparseVector v = new SparseVector(dimension);
      Random rnd = new Random();
      for (int i = 0; i < dimension; i++) {
         v.set(i, rnd.nextGaussian());
      }
      return v;
   }

   /**
    * Static method to create a new <code>SparseVector</code> whose values are 0.
    *
    * @param dimension The dimension of the vector
    * @return a new <code>SparseVector</code> whose values are 0.
    */
   public static Vector zeros(int dimension) {
      return new SparseVector(dimension);
   }

   @Override
   public Vector compress() {
      map.trim();
      return this;
   }

   @Override
   public Vector copy() {
      return new SparseVector(this);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public boolean equals(Object o) {
      return o != null && o instanceof Vector && Arrays.equals(toArray(), Cast.<Vector>as(o).toArray());
   }

   @Override
   public double get(int index) {
      Preconditions.checkPositionIndex(index, dimension());
      return map.get(index);
   }

   @Override
   public int hashCode() {
      return Arrays.hashCode(toArray());
   }

   @Override
   public Vector increment(int index, double amount) {
      if (amount != 0) {
         if (map.containsKey(index)) {
            map.merge(index, amount, Math2::add);
         } else {
            map.put(index, amount);
         }
      }
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
   public Iterator<Vector.Entry> nonZeroIterator() {
      return new Iterator<Vector.Entry>() {
         private final IntIterator indexIter = map.keySet().iterator();

         @Override
         public void forEachRemaining(Consumer<? super Entry> action) {
            while (hasNext()) {
               action.accept(next());
            }
         }

         @Override
         public boolean hasNext() {
            return indexIter.hasNext();
         }

         @Override
         public Vector.Entry next() {
            if (!indexIter.hasNext()) {
               throw new NoSuchElementException();
            }
            int index = indexIter.next();
            return new Vector.Entry(index, get(index));
         }
      };
   }

   @Override
   public Iterator<Vector.Entry> orderedNonZeroIterator() {
      return new Iterator<Vector.Entry>() {
         private final Iterator<Integer> indexIter = map.keySet().stream().sorted().iterator();

         @Override
         public boolean hasNext() {
            return indexIter.hasNext();
         }

         @Override
         public Vector.Entry next() {
            if (!indexIter.hasNext()) {
               throw new NoSuchElementException();
            }
            int index = indexIter.next();
            return new Vector.Entry(index, get(index));
         }
      };
   }

   @Override
   public Vector redim(int newDimension) {
      Vector v = new SparseVector(newDimension);
      for (Iterator<Vector.Entry> itr = nonZeroIterator(); itr.hasNext(); ) {
         Vector.Entry de = itr.next();
         v.set(de.index, de.value);
      }
      return v;
   }

   @Override
   public Vector set(int index, double value) {
      if (value == 0) {
         map.remove(index);
      } else {
         map.put(index, value);
      }
      return this;
   }

   @Override
   public int size() {
      return map.size();
   }

   @Override
   public Vector slice(int from, int to) {
      Preconditions.checkPositionIndex(from, dimension() + 1);
      Preconditions.checkPositionIndex(to, dimension() + 1);
      Preconditions.checkState(to > from, "To index must be > from index");
      SparseVector v = new SparseVector((to - from));
      for (int i = from; i < to; i++) {
         v.set(i - from, get(i));
      }
      return v;
   }

   @Override
   public double[] toArray() {
      final double[] d = new double[dimension()];
      map.int2DoubleEntrySet().forEach(e -> d[e.getIntKey()] = e.getDoubleValue());
      return d;
   }

   @Override
   public String toString() {
      return Arrays.toString(toArray());
   }

   @Override
   public Vector zero() {
      this.map.clear();
      return this;
   }
}//END OF SparseVector
