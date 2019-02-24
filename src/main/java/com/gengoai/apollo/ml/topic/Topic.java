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

import com.gengoai.collection.counter.Counter;

import java.io.Serializable;

/**
 * Defines a topic in a Topic Model
 *
 * @author David B. Bracewell
 */
public class Topic implements Serializable {
   private static final long serialVersionUID = 1L;
   private final int id;
   private final Counter<String> featureDistribution;
   private String name;

   /**
    * Instantiates a new Topic.
    *
    * @param id                  the id
    * @param featureDistribution the feature distribution
    */
   public Topic(int id, Counter<String> featureDistribution) {
      this.id = id;
      this.featureDistribution = featureDistribution;
   }

   /**
    * Gets the feature and their probabilities for a given topic
    *
    * @return the feature distribution
    */
   public Counter<String> getFeatureDistribution() {
      return featureDistribution;
   }

   /**
    * Gets the id of the topic.
    *
    * @return the id
    */
   public int getId() {
      return id;
   }


   /**
    * Gets the name of the topic.
    *
    * @return the name of the topic or null if not one
    */
   public String getName() {
      return name;
   }

   /**
    * Sets the name of the topic.
    *
    * @param name the name of the topic
    */
   public void setName(String name) {
      this.name = name;
   }


}//END OF Topic
