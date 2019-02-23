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

package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.NumericPipeline;
import com.gengoai.apollo.ml.preprocess.Preprocessor;

/**
 * <p>Base regression model that produces a real-value for an input instance.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Regression extends Model {
   private static final long serialVersionUID = 1L;
   private final NumericPipeline modelParameters;

   /**
    * Instantiates a new Regression.
    *
    * @param preprocessors the preprocessors
    */
   public Regression(Preprocessor... preprocessors) {
      this(new NumericPipeline().update(p -> p.preprocessorList.addAll(preprocessors)));
   }

   /**
    * Instantiates a new Regression.
    *
    * @param modelParameters the model parameters
    */
   public Regression(NumericPipeline modelParameters) {
      this.modelParameters = modelParameters;
   }

   @Override
   public int getNumberOfLabels() {
      return 0;
   }

   @Override
   public NumericPipeline getPipeline() {
      return modelParameters;
   }

   /**
    * Estimates a real-value based on the input instance.
    *
    * @param vector the instance
    * @return the estimated value
    */
   public abstract double estimate(Example vector);


}//END OF Regression
