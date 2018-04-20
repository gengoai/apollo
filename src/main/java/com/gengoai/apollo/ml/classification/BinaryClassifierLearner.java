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

package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.conversion.Cast;
import com.gengoai.guava.common.base.Throwables;
import com.gengoai.reflection.Reflect;
import com.gengoai.reflection.ReflectionException;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;

import java.util.Map;

/**
 * <p>Base learner for binary models.</p>
 *
 * @author David B. Bracewell
 */
public abstract class BinaryClassifierLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;

   /**
    * Converts this learner into a multi-class learner using the "one-vs-rest" strategy
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

   @Override
   public BinaryClassifierLearner setParameter(String name, Object value) {
      return Cast.as(super.setParameter(name, value));
   }

   @Override
   public BinaryClassifierLearner setParameters(Map<String, Object> parameters) {
      return Cast.as(super.setParameters(parameters));
   }

   /**
    * Training implementation for binary learner where the "true" label is given
    *
    * @param dataset   the training dataset
    * @param trueLabel the true label
    * @return the classifier
    */
   protected abstract Classifier trainForLabel(Dataset<Instance> dataset, double trueLabel);

   @Override
   protected final Classifier trainImpl(Dataset<Instance> dataset) {
      return trainForLabel(dataset, 1.0);
   }


}//END OF BinaryClassifierLearner
