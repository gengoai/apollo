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

package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.preprocess.Preprocessor;

/**
 * <p>A model whose labels are discrete requiring a {@link com.gengoai.apollo.ml.vectorizer.DiscreteVectorizer} and
 * uses a {@link DiscretePipeline}</p>
 *
 * @author David B. Bracewell
 */
public abstract class DiscreteModel extends Model {
   private static final long serialVersionUID = 1L;
   private final DiscretePipeline modelParameters;

   /**
    * Instantiates a new discrete multi-class model.
    *
    * @param preprocessors the preprocessors
    */
   public DiscreteModel(Preprocessor... preprocessors) {
      this(DiscretePipeline.multiClass().update(p -> p.preprocessorList.addAll(preprocessors)));
   }

   /**
    * Instantiates a new discrete model with the given {@link DiscretePipeline}.
    *
    * @param modelParameters the model parameters
    */
   public DiscreteModel(DiscretePipeline modelParameters) {
      this.modelParameters = modelParameters.copy();
   }

   @Override
   public final int getNumberOfLabels() {
      return modelParameters.labelVectorizer.size();
   }

   @Override
   public final DiscretePipeline getPipeline() {
      return modelParameters;
   }


}//END OF DiscreteModel
