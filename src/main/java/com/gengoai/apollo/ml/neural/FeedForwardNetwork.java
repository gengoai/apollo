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

package com.gengoai.apollo.ml.neural;

import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.classification.Classification;
import com.gengoai.apollo.ml.classification.Classifier;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;

import java.util.ArrayList;

/**
 * @author David B. Bracewell
 */
public class FeedForwardNetwork extends Classifier {
   private static final long serialVersionUID = 1L;
   ArrayList<Layer> layers = new ArrayList<>();

   public FeedForwardNetwork(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   public FeedForwardNetwork(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   @Override
   protected Classifier fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      return this;
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return null;
   }

   @Override
   public Classification predict(Example example) {
      return null;
   }

}//END OF FeedForwardNetwork
