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

import com.davidbracewell.apollo.linalg.Vector;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.Serializable;

/**
 * A collection of <code>HashFunction</code> from the same <code>HashFamily</code>.
 *
 * @author David B. Bracewell
 */
public final class HashGroup implements Serializable {

  private static final long serialVersionUID = 1L;
  private final HashFunction[] hashFunctions;
  private final HashFamily family;


  /**
   * Instantiates a new Hash group.
   *
   * @param family         the hash family
   * @param numberOfHashes the number of hashes
   */
  public HashGroup(HashFamily family, int numberOfHashes) {
    this.family = Preconditions.checkNotNull(family);
    Preconditions.checkArgument(numberOfHashes > 0, "Number of hashes must be >0.");
    hashFunctions = new HashFunction[numberOfHashes];
    for (int i = 0; i < numberOfHashes; i++) {
      hashFunctions[i] = family.create();
    }
  }

  /**
   * Gets the hash family defining the group
   *
   * @return the hash family defining the group
   */
  public HashFamily getFamily() {
    return family;
  }

  /**
   * Gets the number of hash functions in the group
   *
   * @return the number of hash functions in the group
   */
  public int getNumberOfHashes() {
    return hashFunctions.length;
  }

  /**
   * Generates a combined hash code from the individual hash functions in the group.
   *
   * @param v the vector whose hashcode is desired
   * @return the combined hashcode
   */
  public int hash(@NonNull Vector v) {
    int[] hashes = new int[hashFunctions.length];
    for (int i = 0; i < hashFunctions.length; i++) {
      hashes[i] = hashFunctions[i].hash(v);
    }
    return family.combine(hashes);
  }

}//END OF HashGroup
