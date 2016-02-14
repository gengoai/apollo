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
import org.apache.mahout.math.map.OpenIntIntHashMap;

import java.io.Serializable;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.stream.IntStream;

/**
 * @author David B. Bracewell
 */
public class SparseIntVector implements Vector, Serializable {
  private static final long serialVersionUID = 1L;
  private final int dimension;
  private final OpenIntIntHashMap map = new OpenIntIntHashMap();

  public SparseIntVector(int dimension) {
    Preconditions.checkArgument(dimension >= 0);
    this.dimension = dimension;
  }

  @Override
  public Vector compress() {
    map.trimToSize();
    return this;
  }

  @Override
  public Vector copy() {
    SparseIntVector copy = new SparseIntVector(dimension());
    map.forEachPair(copy.map::put);
    return copy;
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
  public Vector increment(int index, double amount) {
    map.adjustOrPutValue(index, (int) amount, (int) amount);
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
  public Iterator<Entry> nonZeroIterator() {
    return new Iterator<Entry>() {
      private final PrimitiveIterator.OfInt indexIter = IntStream.of(map.keys().elements()).iterator();

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

  @Override
  public Iterator<Entry> orderedNonZeroIterator() {
    return new Iterator<Entry>() {
      private final PrimitiveIterator.OfInt indexIter = IntStream.of(map.keys().elements()).sorted().iterator();

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

  @Override
  public Vector set(int index, double value) {
    map.put(index, (int) value);
    return this;
  }

  @Override
  public int size() {
    return map.size();
  }

  @Override
  public Vector slice(int from, int to) {
    SparseIntVector copy = new SparseIntVector(to - from);
    map.forEachPair((i, v) -> {
      if (i >= to && i < from) copy.map.put(i - to, v);
      return true;
    });
    return copy;
  }

  @Override
  public double[] toArray() {
    double[] array = new double[dimension()];
    forEachSparse(e -> array[e.getIndex()] = e.getValue());
    return array;
  }

  @Override
  public Vector zero() {
    map.clear();
    return this;
  }

  @Override
  public Vector redim(int newDimension) {
    SparseIntVector copy = new SparseIntVector(newDimension);
    map.forEachPair((i, v) -> {
      if (i < newDimension) copy.map.put(i, v);
      return true;
    });
    return copy;
  }


}//END OF SparseVector
