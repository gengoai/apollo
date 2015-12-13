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

import com.carrotsearch.hppcrt.cursors.IntDoubleCursor;
import com.carrotsearch.hppcrt.maps.IntDoubleHashMap;
import com.google.common.base.Preconditions;

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
public class SparseVector2 extends AbstractVector {
  private static final long serialVersionUID = 1L;
  private final IntDoubleHashMap map;
  private final int dimension;

  /**
   * Instantiates a new Sparse vector.
   *
   * @param dimension the dimension of the new vector
   */
  public SparseVector2(int dimension) {
    Preconditions.checkArgument(dimension >= 0, "Dimension must be non-negative.");
    this.dimension = dimension;
    this.map = new IntDoubleHashMap();
  }

  /**
   * Copy Constructor
   *
   * @param vector The vector to copy from
   */
  public SparseVector2(Vector vector) {
    this.dimension = Preconditions.checkNotNull(vector.dimension());
    this.map = new IntDoubleHashMap(vector.size());
    for (Iterator<Entry> itr = vector.nonZeroIterator(); itr.hasNext(); ) {
      Entry de = itr.next();
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
    SparseVector2 vector = new SparseVector2(dimension);
    for (int i = 0; i < dimension; i++) {
      vector.map.put(i, 1.0);
    }
    return vector;
  }

  /**
   * Random gaussian vector.
   *
   * @param dimension the dimension
   * @return the vector
   */
  public static Vector randomGaussian(int dimension) {
    SparseVector2 v = new SparseVector2(dimension);
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
    return new SparseVector2(dimension);
  }

  @Override
  public void compress() {

  }

  @Override
  public Vector copy() {
    return new SparseVector2(this);
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public double get(int index) {
    return map.get(index);
  }

  @Override
  public int hashCode() {
    return map.hashCode();
  }

  @Override
  public Vector increment(int index, double amount) {
    map.putOrAdd(index, amount, amount);
    return this;
  }

  @Override
  public boolean isDense() {
    return false;
  }

  @Override
  public Iterator<Entry> nonZeroIterator() {
    return new SparseIterator();
  }

  @Override
  public Iterator<Entry> orderedNonZeroIterator() {
    return new OrderedSparseIterator();
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
    Preconditions.checkPositionIndex(from, dimension());
    Preconditions.checkPositionIndex(to, dimension() + 1);
    Preconditions.checkState(to > from, "To index must be > from index");
    SparseVector2 v = new SparseVector2((to - from));
    for (int i = from; i < to; i++) {
      v.set(i, get(i));
    }
    return v;
  }

  @Override
  public double[] toArray() {
    final double[] d = new double[dimension()];
    map.forEach((Consumer<IntDoubleCursor>) intDoubleCursor -> {
      d[intDoubleCursor.key] = intDoubleCursor.value;
    });
    return d;
  }

  @Override
  public String toString() {
    return map.toString();
  }

  @Override
  public Vector zero() {
    this.map.clear();
    return this;
  }

  @Override
  public Vector redim(int newDimension) {
    Vector v = new SparseVector2(newDimension);
    for (Iterator<Entry> itr = nonZeroIterator(); itr.hasNext(); ) {
      Entry de = itr.next();
      v.set(de.index, de.value);
    }
    return v;
  }

  private class SparseIterator implements Iterator<Entry>, Serializable {

    private static final long serialVersionUID = 1L;
    /**
     * The Index.
     */
    int index = 0;
    private int[] indexes = map.keys;
    private double[] values = map.values;


    @Override
    public boolean hasNext() {
      return index < indexes.length;
    }

    @Override
    public Entry next() {
      if (index >= indexes.length) {
        throw new NoSuchElementException();
      }
      int ii = indexes[index];
      Entry de = new Entry(ii, values[index]);
      index++;
      return de;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }//END OF SparseIterator

  private class OrderedSparseIterator implements Iterator<Entry>, Serializable {

    private static final long serialVersionUID = 1L;
    /**
     * The Index.
     */
    int index = 0;
    private int[] indexes = map.keys;
    private double[] values = map.values;


    /**
     * Instantiates a new Ordered sparse iterator.
     */
    OrderedSparseIterator() {
      Arrays.sort(indexes);
    }


    @Override
    public boolean hasNext() {
      return index < indexes.length;
    }

    @Override
    public Entry next() {
      if (index >= indexes.length) {
        throw new NoSuchElementException();
      }
      int ii = indexes[index];
      Entry de = new Entry(ii, values[index]);
      index++;
      return de;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }//END OF SparseIterator

}//END OF SparseVector
