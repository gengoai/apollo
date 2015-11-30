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

package com.davidbracewell.apollo.learning;

import lombok.NonNull;
import lombok.Value;

import java.io.Serializable;

/**
 * The type Feature.
 *
 * @author David B. Bracewell
 */
@Value
public class Feature implements Serializable, Comparable<Feature> {
  private static final long serialVersionUID = 1L;
  private String name;
  private double value;

  private Feature(String name, double value) {
    this.name = name;
    this.value = value;
  }

  /**
   * Boolean feature feature.
   *
   * @param name  the name
   * @param value the value
   * @return the feature
   */
  public static Feature binary(@NonNull String name, boolean value) {
    return new Feature(name, value ? 1.0 : 0.0);
  }

  /**
   * True feature.
   *
   * @param name the name
   * @return the feature
   */
  public static Feature TRUE(@NonNull String name) {
    return new Feature(name, 1.0);
  }

  /**
   * False feature.
   *
   * @param name the name
   * @return the feature
   */
  public static Feature FALSE(@NonNull String name) {
    return new Feature(name, 0.0);
  }

  /**
   * Real feature feature.
   *
   * @param name  the name
   * @param value the value
   * @return the feature
   */
  public static Feature real(@NonNull String name, double value) {
    return new Feature(name, value);
  }


  @Override
  public int compareTo(Feature o) {
    return o == null ? 1 : this.name.compareTo(o.name);
  }
}//END OF Feature
