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
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.reflection.Reflect;
import com.davidbracewell.reflection.ReflectionException;
import com.davidbracewell.stream.MStream;
import com.google.common.base.Throwables;

import java.util.Map;

/**
 * The type Binary classifier learner.
 *
 * @author David B. Bracewell
 */
public abstract class BinaryClassifierLearner extends ClassifierLearner {
  private static final long serialVersionUID = 1L;

  @Override
  protected Classifier trainImpl(Dataset<Instance> dataset) {
    return trainFromStream(
      () -> dataset.stream().map(i -> i.toVector(dataset.getEncoderPair())),
      dataset.getEncoderPair(),
      dataset.getPreprocessors()
    );
  }

  /**
   * Train from supplier classifier.
   *
   * @param supplier      the supplier
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   * @return the classifier
   */
  protected abstract Classifier trainFromStream(
    SerializableSupplier<MStream<FeatureVector>> supplier,
    EncoderPair encoderPair,
    PreprocessorList<Instance> preprocessors
  );


  /**
   * One vs rest classifier learner.
   *
   * @return the classifier learner
   */
  public final ClassifierLearner oneVsRest() {
    final Map<String, Object> parameters = Cast.as(getParameters());
    final Class<? extends BinaryClassifierLearner> clazz = this.getClass();
    return new OneVsRestLearner(() -> {
      BinaryClassifierLearner learner = null;
      try {
        learner = Reflect.onClass(clazz).create().get();
        learner.setParameters(parameters);
        return learner;
      } catch (ReflectionException e) {
        throw Throwables.propagate(e);
      }
    }
    );
  }


}//END OF BinaryClassifierLearner
