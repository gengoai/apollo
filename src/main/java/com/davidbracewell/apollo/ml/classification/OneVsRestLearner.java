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
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.util.stream.IntStream;

/**
 * @author David B. Bracewell
 */
public class OneVsRestLearner extends ClassifierLearner {
  private static final long serialVersionUID = 1L;
  private final SerializableSupplier<BinaryClassifierLearner> learnerSupplier;

  public OneVsRestLearner(@NonNull SerializableSupplier<BinaryClassifierLearner> learnerSupplier) {
    this.learnerSupplier = learnerSupplier;
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
          bcl.reset();
          SerializableSupplier<MStream<FeatureVector>> supplier = () -> dataset.stream().map(instance -> {
            FeatureVector vector = instance.toVector(dataset.getEncoderPair());
            if (dataset.getLabelEncoder().encode(instance.getLabel()) == i) {
              vector.setLabel(1);
            } else {
              vector.setLabel(0);
            }
            return vector;
          });
          return bcl.trainFromStream(
            supplier,
            dataset.getEncoderPair(),
            dataset.getPreprocessors()
          );
        }
      ).toArray(Classifier[]::new);

    return model;
  }


  @Override
  public void reset() {

  }

}//END OF OneVsRestLearner
