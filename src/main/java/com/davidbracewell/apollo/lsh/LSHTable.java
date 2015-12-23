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

import java.io.Serializable;
import java.util.Set;

/**
 * The type LSH table.
 *
 * @author David B. Bracewell
 */
public abstract class LSHTable implements Serializable {
  private static final long serialVersionUID = 1L;

  private final HashGroup hashGroup;

  /**
   * Instantiates a new LSH table.
   *
   * @param hashGroup the hash group
   */
  protected LSHTable(HashGroup hashGroup) {
    this.hashGroup = hashGroup;
  }

  /**
   * Add void.
   *
   * @param input the input
   */
  public final void add(final Vector input, int index) {
    add(hashGroup.hash(input), index);
  }

  /**
   * Add void.
   *
   * @param hash  the hash
   * @param index the index of the item being hashed
   */
  protected abstract void add(final int hash, int index);

  /**
   * Get list.
   *
   * @param hash the hash
   * @return the list
   */
  protected abstract Set<Integer> get(final int hash);

  /**
   * Get list.
   *
   * @param input the input
   * @return the list
   */
  public final Set<Integer> get(final Vector input) {
    return get(hashGroup.hash(input));
  }


  public HashGroup getHashGroup() {
    return hashGroup;
  }

}//END OF LSHTable
