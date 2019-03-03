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

import com.gengoai.apollo.ml.params.ParamValuePair;
import com.gengoai.apollo.optimization.WeightUpdate;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.loss.LossFunction;

/**
 * @author David B. Bracewell
 */
public interface OptimizableParamSet {
   static ParamValuePair<Activation> activation(Activation activation) {
      return Model.activation.set(activation);
   }

   static ParamValuePair<LossFunction> activation(LossFunction lossFunction) {
      return Model.lossFunction.set(lossFunction);
   }

   static ParamValuePair<Integer> batchSize(int batchSize) {
      return Model.batchSize.set(batchSize);
   }

   static ParamValuePair<Boolean> cacheData(Boolean cacheData) {
      return Model.cacheData.set(cacheData);
   }

   static ParamValuePair<Integer> historySize(int historySize) {
      return Model.historySize.set(historySize);
   }

   static ParamValuePair<Integer> maxIterations(int iterations) {
      return Model.maxIterations.set(iterations);
   }

   static ParamValuePair<Integer> reportInterval(int reportInterval) {
      return Model.reportInterval.set(reportInterval);
   }

   static ParamValuePair<Double> tolerance(Double tolerance) {
      return Model.tolerance.set(tolerance);
   }

   static ParamValuePair<WeightUpdate> weightUpdater(WeightUpdate weightUpdater) {
      return Model.weightUpdater.set(weightUpdater);
   }


}//END OF OptimizableParamSet
