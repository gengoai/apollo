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

import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * A sparse vector implementation backed by a map
 *
 * @author David B. Bracewell
 */
public class SparseVector extends BaseVector {
   private static final long serialVersionUID = 1L;
   private final OpenIntDoubleHashMap map;
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
      this.map = new OpenIntDoubleHashMap();
   }

   /**
    * Copy Constructor
    *
    * @param vector The vector to copy from
    */
   public SparseVector(@NonNull Vector vector) {
      this.dimension = vector.dimension();
      this.map = new OpenIntDoubleHashMap(vector.size());
      for (Iterator<Vector.Entry> itr = vector.nonZeroIterator(); itr.hasNext(); ) {
         Vector.Entry de = itr.next();
         this.map.put(de.index, de.value);
      }
      setLabel(vector.getLabel());
      setWeight(vector.getWeight());
      setPredicted(vector.getPredicted());
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
      map.trimToSize();
      return this;
   }

   @Override
   public Vector copy() {
      return new SparseVector(this)
                .setLabel(getLabel())
                .setWeight(getWeight())
                .setPredicted(getPredicted());
   }

   @Override
   protected Vector createNew(int dimension) {
      return new SparseVector(dimension);
   }


   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public double get(int index) {
      Preconditions.checkPositionIndex(index, dimension());
      return map.get(index);
   }

   @Override
   public Vector increment(int index, double amount) {
      if (amount != 0) {
         map.adjustOrPutValue(index, amount, amount);
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
      return new itr(false);
   }

   @Override
   public Iterator<Vector.Entry> orderedNonZeroIterator() {
      return new itr(true);
   }

   @Override
   public Vector set(int index, double value) {
      if (value == 0) {
         map.removeKey(index);
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
   public Vector zero() {
      this.map.clear();
      return this;
   }

   private class itr implements Iterator<Vector.Entry> {
      private final IntArrayList keys;
      private int index = 0;

      public itr(boolean sorted) {
         keys = map.keys();
         if (sorted) {
            keys.sort();
         }
      }

      @Override
      public boolean hasNext() {
         return index < keys.size();
      }

      @Override
      public Entry next() {
         if (index >= keys.size()) {
            throw new NoSuchElementException();
         }
         int key = keys.get(index);
         index++;
         double value = get(key);
         return new Vector.Entry(key, value);
      }
   }

}//END OF SparseVector
