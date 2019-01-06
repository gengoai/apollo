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

package com.gengoai.apollo.ml.topic;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.ModelParameters;
import com.gengoai.apollo.ml.preprocess.Preprocessor;

/**
 * <p>A model to estimates topics for examples.</p>
 *
 * @author David B. Bracewell
 */
public abstract class TopicModel extends Model<TopicModel> {
   private static final long serialVersionUID = 1L;

   public TopicModel(Preprocessor...preprocessors) {
      this(ModelParameters.indexedLabelVectorizer().preprocessors(preprocessors));
   }

   public TopicModel(ModelParameters modelParameters) {
      super(modelParameters);
   }

   /**
    * Estimates the distribution of topics for the given example
    *
    * @param Example the example to estimate topics for
    * @return An NDArray with topic scores
    */
   public abstract double[] estimate(Example Example);

   /**
    * Gets the distribution across topics for a given feature.
    *
    * @param feature the feature (word) whose topic distribution is desired
    * @return the distribution across topics for the given feature
    */
   public abstract NDArray getTopicDistribution(String feature);


   public abstract Topic getTopic(int topic);


   public abstract int getNumberOfTopics();

}//END OF TopicModel
