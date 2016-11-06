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

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.conversion.Val;
import com.davidbracewell.function.SerializableSupplier;
import lombok.NonNull;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The type One vs rest learner.
 *
 * @author David B. Bracewell
 */
public class OneVsRestLearner extends ClassifierLearner {
  private static final long serialVersionUID = 1L;
  private final Map<String, Object> parameters = new HashMap<>();
  private volatile SerializableSupplier<BinaryClassifierLearner> learnerSupplier;
  private boolean normalize = false;

  /**
   * Instantiates a new One vs rest learner.
   *
   * @param learnerSupplier the learner supplier
   */
  public OneVsRestLearner(@NonNull SerializableSupplier<BinaryClassifierLearner> learnerSupplier) {
    this.learnerSupplier = learnerSupplier;
    this.parameters.putAll(learnerSupplier.get().getParameters());
    this.parameters.put("binaryLearner", learnerSupplier.get().getClass());
    this.parameters.put("normalize", false);
  }

  @Override
  protected Classifier trainImpl(Dataset<Instance> dataset) {
    OneVsRestClassifier model = new OneVsRestClassifier(
      dataset.getEncoderPair(),
      dataset.getPreprocessors()
    );
    model.classifiers = IntStream.range(0, dataset.getLabelEncoder().size())
      .parallel()
      .mapToObj(i -> {
          BinaryClassifierLearner bcl = learnerSupplier.get();
          bcl.setParameters(parameters);
          bcl.reset();
          return bcl.trainForLabel(dataset, i);
        }
      ).toArray(Classifier[]::new);

    return model;
  }


  @Override
  public void reset() {

  }

  @Override
  public Map<String, ?> getParameters() {
    return parameters;
  }

  @Override
  public void setParameters(@NonNull Map<String, Object> parameters) {
    if (parameters.containsKey("binaryLearner")) {
      final BinaryClassifierLearner learner = Val.of(parameters.get("binaryLearner")).as(BinaryClassifierLearner.class);
      this.learnerSupplier = () -> learner;
    }
    if (parameters.containsKey("normalize")) {
      this.normalize = Val.of(parameters.get("normalize")).asBooleanValue();
    }
    this.parameters.clear();
    this.parameters.putAll(parameters);
  }

  @Override
  public void setParameter(String name, Object value) {
    if (name.equals("binaryLearner")) {
      final BinaryClassifierLearner learner = Cast.as(value);
      this.learnerSupplier = () -> learner;
    }
    if (name.equals("normalize")) {
      this.normalize = Val.of(value).asBooleanValue();
    }
    parameters.put(name, value);
  }

  @Override
  public Object getParameter(String name) {
    if (name.equals("normalize")) {
      return this.normalize;
    } else if (name.equals("binaryLearner")) {
      return learnerSupplier.get();
    }
    return parameters.get(name);
  }


}//END OF OneVsRestLearner
