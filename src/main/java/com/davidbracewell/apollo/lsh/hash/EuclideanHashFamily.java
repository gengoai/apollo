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

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.affinity.EuclideanDistance;
import com.davidbracewell.apollo.lsh.HashFamily;
import com.davidbracewell.apollo.lsh.HashFunction;
import com.google.common.base.Preconditions;

import java.util.Arrays;

/**
 * Hash family good for indexing by euclidean distance.
 *
 * @author David B. Bracewell
 */
public class EuclideanHashFamily implements HashFamily {

  private static final long serialVersionUID = 1L;
  private final int dimension;
  private final int w;

  /**
   * Instantiates a new Euclidean hash family.
   *
   * @param dimension the dimension of the random projection vector, i.e. number of elements in the vectors being
   *                  indexed.
   * @param w         the w
   */
  public EuclideanHashFamily(int dimension, int w) {
    Preconditions.checkArgument(dimension > 0, "Dimension must be >0");
    Preconditions.checkArgument(w > 0, "w must be >0");
    this.dimension = dimension;
    this.w = w;
  }

  @Override
  public int combine(int[] hashes) {
    return Arrays.hashCode(hashes);
  }

  @Override
  public HashFunction create() {
    return new EuclideanHashFunction(dimension, w);
  }

  @Override
  public DistanceMeasure getDistanceMeasure() {
    return new EuclideanDistance();
  }

}//END OF EuclideanHashFamily
