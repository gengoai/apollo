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

package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.reflection.BeanMap;
import com.davidbracewell.reflection.Ignore;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Map;

/**
 * The interface Classifier learner.
 *
 * @author David B. Bracewell
 */
public abstract class ClassifierLearner implements Serializable {
  private static final long serialVersionUID = 1L;

  /**
   * Builder learner builder.
   *
   * @return the learner builder
   */
  public static LearnerBuilder builder() {
    return new LearnerBuilder();
  }

  /**
   * Train classifier.
   *
   * @param dataset the dataset
   * @return the classifier
   */
  public final Classifier train(@NonNull Dataset<Instance> dataset) {
    Classifier classifier = trainImpl(dataset);
    classifier.finishTraining();
    return classifier;
  }

  protected abstract Classifier trainImpl(Dataset<Instance> dataset);

  /**
   * Gets parameters.
   *
   * @return the parameters
   */
  @Ignore
  public Map<String, ?> getParameters() {
    return new BeanMap(this);
  }

  /**
   * Sets parameters.
   *
   * @param parameters the parameters
   */
  @Ignore
  public void setParameters(@NonNull Map<String, Object> parameters) {
    new BeanMap(this).putAll(parameters);
  }

  /**
   * Sets parameter.
   *
   * @param name  the name
   * @param value the value
   */
  @Ignore
  public void setParameter(String name, Object value) {
    new BeanMap(this).put(name, value);
  }

  /**
   * Gets parameter.
   *
   * @param name the name
   * @return the parameter
   */
  @Ignore
  public Object getParameter(String name) {
    return new BeanMap(this).get(name);
  }


}//END OF ClassifierLearner
