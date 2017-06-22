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
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * <p>Learner that learns K binary classifiers, where K is the number of labels, and produces a multi-class model.</p>
 *
 * @author David B. Bracewell
 */
public class OneVsRestLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;
   private final Map<String, Object> parameters = new HashMap<>();
   private volatile SerializableSupplier<BinaryClassifierLearner> learnerSupplier;
   @Getter
   @Setter
   private boolean normalize = true;

   /**
    * Instantiates a new One vs rest learner.
    */
   public OneVsRestLearner() {
      this(AveragedPerceptronLearner::new);
   }

   /**
    * Instantiates a new One vs rest learner.
    *
    * @param learnerSupplier the binary classifier learner supplier
    */
   public OneVsRestLearner(@NonNull SerializableSupplier<BinaryClassifierLearner> learnerSupplier) {
      this.learnerSupplier = learnerSupplier;
      this.parameters.putAll(learnerSupplier.get().getParameters());
      this.parameters.put("binaryLearner", learnerSupplier.get().getClass());
      this.parameters.put("normalize", false);
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

   @Override
   public Map<String, ?> getParameters() {
      return parameters;
   }

   @Override
   public OneVsRestLearner setParameters(@NonNull Map<String, Object> parameters) {
      this.parameters.clear();
      parameters.forEach(this::setParameter);
      return Cast.as(this);
   }

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   public OneVsRestLearner setParameter(String name, Object value) {
      if (name.equals("binaryLearner")) {
         this.learnerSupplier = () -> Val.of(parameters.get("binaryLearner"))
                                         .as(BinaryClassifierLearner.class);
      }
      if (name.equals("normalize")) {
         this.normalize = Val.of(value).asBooleanValue();
      }
      parameters.put(name, value);
      return this;
   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      OneVsRestClassifier model = new OneVsRestClassifier(this);
      model.classifiers = IntStream.range(0, dataset.getLabelEncoder().size())
                                   .parallel()
                                   .mapToObj(i -> {
                                                BinaryClassifierLearner bcl = learnerSupplier.get();
                                                bcl.setParameters(parameters);
                                                bcl.reset();
                                                bcl.update(dataset.getEncoderPair(), dataset.getPreprocessors(), dataset.getVectorizer());
                                                return bcl.trainForLabel(dataset, i);
                                             }
                                            ).toArray(Classifier[]::new);
      model.normalize = normalize;
      return model;
   }


}//END OF OneVsRestLearner
