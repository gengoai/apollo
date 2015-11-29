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

import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.ToString;

import java.io.Serializable;
import java.util.Set;

/**
 * The type Instance.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode
@ToString
public class Instance implements Serializable {
  private static final long serialVersionUID = 1L;
  private final Set<Feature> features;
  private Object label;

  /**
   * Instantiates a new Instance.
   *
   * @param features the features
   */
  public Instance(@NonNull Set<Feature> features) {
    this.features = features;
  }

  /**
   * Instantiates a new Instance.
   *
   * @param features the features
   * @param label    the label
   */
  public Instance(@NonNull Set<Feature> features, Object label) {
    this.features = features;
    this.label = label;
  }

  /**
   * Gets label.
   *
   * @return the label
   */
  public Object getLabel() {
    return label;
  }

  /**
   * Sets label.
   *
   * @param label the label
   */
  public void setLabel(Object label) {
    this.label = label;
  }

  /**
   * Has label boolean.
   *
   * @return the boolean
   */
  public boolean hasLabel() {
    return label != null;
  }


  /**
   * Create instance.
   *
   * @param features the features
   * @return the instance
   */
  public static Instance create(@NonNull Set<Feature> features) {
    return new Instance(features);
  }

  /**
   * Create instance.
   *
   * @param features the features
   * @param label    the label
   * @return the instance
   */
  public static Instance create(@NonNull Set<Feature> features, Object label) {
    return new Instance(features, label);
  }

}//END OF Instance
