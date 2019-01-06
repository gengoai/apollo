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

package com.gengoai.apollo.ml.preprocess;


import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;

/**
 * <p>Preprocessors represent filters and transforms to apply to a dataset before building a model. This allows such
 * things as removing bad values or features, feature selection, and value normalization. </p>
 *
 * @author David B. Bracewell
 */
public interface Preprocessor {

   /**
    * Applies the transform tho the given example.
    *
    * @param example the example to apply the transform to
    * @return A new example with the transformed applied
    */
   Example apply(Example example);


   /**
    * Determines the parameters, e.g. counts, etc., of the preprocessor from the given dataset. Implementations should
    * relearn parameters on each call instead of updating.
    *
    * @param dataset the dataset to fit this preprocessors parameters to
    */
   Dataset fitAndTransform(Dataset dataset);


   /**
    * Resets the parameters of the preprocessor.
    */
   void reset();


}//END OF Preprocessor
