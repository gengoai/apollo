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

package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.statistics.measure.Measure;
import com.gengoai.stream.MStream;

class DummyFlatClusterer extends FlatCentroidClusterer {
   private static final long serialVersionUID = 1L;


   public DummyFlatClusterer(DiscretePipeline modelParameters, Measure measure) {
      super(modelParameters);
      setMeasure(measure);
   }


   @Override
   public void fit(MStream<NDArray> vectors, ParamMap fitParameters) {
      throw new UnsupportedOperationException();
   }

   @Override
   public ParamMap getDefaultFitParameters() {
      throw new UnsupportedOperationException();
   }

}//END OF DummyFlatClusterer
