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

import com.gengoai.apollo.ml.vectorizer.*;

/**
 * <p>A pipeline for models whose labels are discrete values, i.e. strings</p>
 *
 * @author David B. Bracewell
 */
public class DiscretePipeline extends Pipeline<DiscreteVectorizer, DiscretePipeline> {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new DiscretePipeline.
    *
    * @param labelVectorizer the label vectorizer
    */
   protected DiscretePipeline(DiscreteVectorizer labelVectorizer) {
      super(labelVectorizer);
   }


   /**
    * Creates a new DiscretePipeline with a {@link BinaryLabelVectorizer} for labels that <code>true</code> or
    * <code>false</code>.
    *
    * @return the DiscretePipeline
    */
   public static DiscretePipeline binary() {
      return new DiscretePipeline(BinaryLabelVectorizer.INSTANCE);
   }

   /**
    * Creates a new  DiscretePipeline for multi-class problems using an {@link IndexVectorizer} as the label
    * vectorizer.
    *
    * @return the DiscretePipeline
    */
   public static DiscretePipeline multiclass() {
      return new DiscretePipeline(new MultiLabelBinarizer());
   }

   /**
    * Creates a new DiscretePipeline for binary or multi-class problems
    *
    * @param isBinary True - create a pipeline for binary problems, False create a pipeline for multi-class problems.
    * @return the DiscretePipeline
    */
   public static DiscretePipeline create(boolean isBinary) {
      return isBinary ? binary() : multiclass();
   }

   /**
    * Creates a new DiscretePipeline for unsupervised problems using a {@link NoOptVectorizer} as the label vectorizer.
    *
    * @return the DiscretePipeline
    */
   public static DiscretePipeline unsupervised() {
      return create(NoOptVectorizer.INSTANCE);
   }

   /**
    * Creates a new DiscretePipeline with the given label vectorizer
    *
    * @param labelVectorizer the label vectorizer
    * @return the DiscretePipeline
    */
   public static DiscretePipeline create(DiscreteVectorizer labelVectorizer) {
      return new DiscretePipeline(labelVectorizer);
   }


}//END OF DiscretePipeline
