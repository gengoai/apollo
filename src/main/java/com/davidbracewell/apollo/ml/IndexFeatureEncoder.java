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

package com.davidbracewell.apollo.ml;

import com.davidbracewell.collection.Index;
import com.davidbracewell.collection.Indexes;

import java.io.Serializable;
import java.util.Collection;
import java.util.Collections;

/**
 * @author David B. Bracewell
 */
public class IndexFeatureEncoder implements FeatureEncoder, Serializable {
  private static final long serialVersionUID = 1L;
  private volatile Index<String> index = Indexes.newIndex();
  private boolean frozen = false;

  @Override
  public String decode(int value) {
    return index.get(value);
  }

  @Override
  public int encode(String feature) {
    if (!index.contains(feature) && !frozen) {
      synchronized (this) {
        if (!index.contains(feature)) {
          index.add(feature);
        }
      }
    }
    return index.indexOf(feature);
  }

  @Override
  public Collection<String> features() {
    return Collections.unmodifiableList(index.asList());
  }

  @Override
  public void freeze() {
    frozen = true;
  }

  @Override
  public void unFreeze() {
    frozen = false;
  }

  @Override
  public boolean isFrozen() {
    return frozen;
  }

  @Override
  public int size() {
    return index.size();
  }

}//END OF IndexFeatureEncoder
