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
 *
 */

package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.DiscreteModel;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Pipeline;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.function.SerializableFunction;

/**
 * <p>
 * A classifier assigns one or more labels (categories) to an input object. The input object, or {@link
 * com.gengoai.apollo.ml.Instance}*, is described using one or more {@link com.gengoai.apollo.ml.Feature}s describing
 * the salient characteristics of the input.
 * </p>
 * <p>
 * Classification is typically done in a supervised manner where the classifier produces a function that converts an
 * given input {@link Example} to a {@link Classification} result describing the scores for the discrete labels in the
 * problem.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class Classifier extends DiscreteModel implements SerializableFunction<Example, Classification> {
   private static final long serialVersionUID = 1L;


   /**
    * Instantiates a new Classifier.
    *
    * @param preprocessors the preprocessors
    */
   public Classifier(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Classifier with the given {@link Pipeline}.
    *
    * @param modelParameters the model parameters
    */
   public Classifier(DiscretePipeline modelParameters) {
      super(modelParameters);
   }


   @Override
   public Classification apply(Example example) {
      return predict(example);
   }

   /**
    * <p>Predicts the label(s) for a given example encoding and preprocessing the example as needed.</p>
    *
    * @param example the example
    * @return the classification result
    */
   public abstract Classification predict(Example example);

}//END OF Classifier
