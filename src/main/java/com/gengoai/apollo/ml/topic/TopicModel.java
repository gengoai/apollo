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

import com.gengoai.apollo.linear.p2.NDArray;
import com.gengoai.apollo.ml.DiscreteModel;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.preprocess.Preprocessor;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>A model to estimates topics for examples.</p>
 *
 * @author David B. Bracewell
 */
public abstract class TopicModel extends DiscreteModel {
   private static final long serialVersionUID = 1L;
   /**
    * The Topics.
    */
   protected final List<Topic> topics = new ArrayList<>();

   /**
    * Instantiates a new Topic model.
    *
    * @param preprocessors the preprocessors
    */
   public TopicModel(Preprocessor... preprocessors) {
      super(DiscretePipeline.unsupervised().update(p -> p.preprocessorList.addAll(preprocessors)));
   }

   /**
    * Instantiates a new Topic model.
    *
    * @param modelParameters the model parameters
    */
   public TopicModel(DiscretePipeline modelParameters) {
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


   /**
    * Gets the topic by its id.
    *
    * @param id the id of the topic to retrieve
    * @return the topic
    * @throws IndexOutOfBoundsException if the id is invalid
    */
   public Topic getTopic(int id) {
      return topics.get(id);
   }

   /**
    * Gets the topic by its name.
    *
    * @param name the name of the topic
    * @return the topic
    * @throws IndexOutOfBoundsException if the name is invalid
    */
   public Topic getTopic(String name) {
      return topics.stream()
                   .filter(t -> name.equals(t.getName()))
                   .findFirst()
                   .orElseThrow(IndexOutOfBoundsException::new);
   }


   /**
    * Gets number of topics.
    *
    * @return the number of topics
    */
   public int size() {
      return topics.size();
   }

}//END OF TopicModel
