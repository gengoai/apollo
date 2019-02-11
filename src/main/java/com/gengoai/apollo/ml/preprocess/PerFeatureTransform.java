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
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.vectorizer.CountFeatureVectorizer;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.function.SerializableFunction;
import com.gengoai.string.Strings;

import java.io.Serializable;

/**
 * <p>A preprocessor that generates one preprocessor of a given type per feature prefix in the dataset. </p>
 *
 * @author David B. Bracewell
 */
public class PerFeatureTransform implements Preprocessor, Serializable {
   private static final long serialVersionUID = 1L;
   private final PreprocessorList preprocessors = new PreprocessorList();
   private final SerializableFunction<String, Preprocessor> preprocessorSupplier;

   /**
    * Instantiates a new Per feature transform.
    *
    * @param preprocessorSupplier the preprocessor supplier
    */
   public PerFeatureTransform(SerializableFunction<String, Preprocessor> preprocessorSupplier) {
      this.preprocessorSupplier = preprocessorSupplier;
   }

   @Override
   public Example apply(Example example) {
      return preprocessors.apply(example);
   }

   @Override
   public Dataset fitAndTransform(Dataset dataset) {
      IndexVectorizer features = new CountFeatureVectorizer();
      features.fit(dataset);
      features.alphabet()
              .stream()
              .map(Feature::getPrefix)
              .distinct()
              .filter(Strings::isNotNullOrBlank)
              .map(preprocessorSupplier)
              .forEach(preprocessors::add);
      return preprocessors.fitAndTransform(dataset);
   }

   @Override
   public void reset() {
      preprocessors.clear();
   }


}//END OF PerFeatureTransform
