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

package com.davidbracewell.apollo.lsh;


import com.davidbracewell.apollo.similarity.DistanceMeasure;

import java.io.Serializable;

/**
 * Defines a family of hash functions.
 *
 * @author David B. Bracewell
 */
public interface HashFamily extends Serializable {

  /**
   * Combines a number of individual hashes into one
   *
   * @param hashes the hashes to combine
   * @return the combined hash value
   */
  int combine(int[] hashes);

  /**
   * Creates a new hash function in the family.
   *
   * @return a new hash function
   */
  HashFunction create();

  /**
   * Gets the distance measure associated with the hash family.
   *
   * @return the distance measure associated with the hash family
   */
  DistanceMeasure getDistanceMeasure();

}//END OF HashFamily
