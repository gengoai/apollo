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

package com.davidbracewell.apollo.lsh.hash;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.lsh.HashFunction;
import com.google.common.base.Preconditions;

/**
 * A hash function designed to make items have the same hash value if their euclidean distance is small.
 *
 * @author David B. Bracewell
 */
public class EuclideanHashFunction implements HashFunction {
  private static final long serialVersionUID = 1L;

  private final Vector random;
  private final int offset;
  private final int w;

  /**
   * Instantiates a new Euclidean hash function.
   *
   * @param dimension the dimension of the random projection vector, i.e. number of elements in the vectors being
   *                  indexed.
   * @param w         the w
   */
  public EuclideanHashFunction(int dimension, int w) {
    Preconditions.checkArgument(dimension > 0, "Dimension must be >0");
    this.random = SparseVector.randomGaussian(dimension);
    this.offset = (int) Math.floor(Math.random() * w);
    this.w = w;
  }

  @Override
  public int hash(Vector vector) {
    return (int) Math.round((vector.dot(random) + offset) / (double) w);
  }

}//END OF EuclideanHashFunction
