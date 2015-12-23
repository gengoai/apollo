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

import com.davidbracewell.apollo.affinity.CosineSimilarity;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.affinity.OneMinusSimilarityDistance;
import com.davidbracewell.apollo.lsh.HashFamily;
import com.davidbracewell.apollo.lsh.HashFunction;
import com.google.common.base.Preconditions;

/**
 * A hash family good for dealing with cosine distances.
 *
 * @author David B. Bracewell
 */
public class CosineHashFamily implements HashFamily {

  private static final long serialVersionUID = 1L;
  private final int dimension;

  /**
   * Instantiates a new Cosine hash family.
   *
   * @param dimension the dimension of the random projection vector, i.e. number of elements in the vectors being
   *                  indexed.
   */
  public CosineHashFamily(int dimension) {
    Preconditions.checkArgument(dimension > 0, "Dimension must be >0.");
    this.dimension = dimension;
  }

  @Override
  public int combine(int[] hashes) {
    int pow = 1;
    int result = 0;
    for (int hash : hashes) {
      result += hash == 0 ? 0 : pow;
      pow *= 2;
    }
    return result;
  }

  @Override
  public HashFunction create() {
    return new CosineHashFunction(dimension);
  }

  @Override
  public DistanceMeasure getDistanceMeasure() {
    return new OneMinusSimilarityDistance(new CosineSimilarity());
  }

}//END OF CosineHashFamily
