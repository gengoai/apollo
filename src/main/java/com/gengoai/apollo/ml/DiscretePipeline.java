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

import com.gengoai.Copyable;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.vectorizer.*;

/**
 * <p>A pipeline for models whose labels are discrete values, i.e. strings</p>
 *
 * @author David B. Bracewell
 */
public class DiscretePipeline extends Pipeline<DiscreteVectorizer, DiscretePipeline> implements Copyable<DiscretePipeline> {
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
    * @param preprocessors the preprocessors
    * @return the DiscretePipeline
    */
   public static DiscretePipeline binary(Preprocessor... preprocessors) {
      return binary(new CountFeatureVectorizer(), preprocessors);
   }

   /**
    * Creates a new DiscretePipeline with a {@link BinaryLabelVectorizer} for labels that <code>true</code> or
    * <code>false</code>.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    * @return the DiscretePipeline
    */
   public static DiscretePipeline binary(DiscreteVectorizer featureVectorizer, Preprocessor... preprocessors) {
      DiscretePipeline pipeline = new DiscretePipeline(BinaryLabelVectorizer.INSTANCE);
      pipeline.featureVectorizer = featureVectorizer;
      pipeline.preprocessorList.addAll(preprocessors);
      return pipeline;
   }


   /**
    * Creates a new  DiscretePipeline for multi-class problems using an {@link IndexVectorizer} as the label
    * vectorizer.
    *
    * @param preprocessors the preprocessors
    * @return the DiscretePipeline
    */
   public static DiscretePipeline multiClass(Preprocessor... preprocessors) {
      return multiClass(new CountFeatureVectorizer(), preprocessors);
   }

   /**
    * Creates a new  DiscretePipeline for multi-class problems using an {@link IndexVectorizer} as the label
    * vectorizer.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    * @return the DiscretePipeline
    */
   public static DiscretePipeline multiClass(DiscreteVectorizer featureVectorizer, Preprocessor... preprocessors) {
      DiscretePipeline pipeline = new DiscretePipeline(new MultiLabelBinarizer());
      pipeline.featureVectorizer = featureVectorizer;
      pipeline.preprocessorList.addAll(preprocessors);
      return pipeline;
   }


   /**
    * Creates a new DiscretePipeline for binary or multi-class problems
    *
    * @param isBinary True - create a pipeline for binary problems, False create a pipeline for multi-class problems.
    * @return the DiscretePipeline
    */
   public static DiscretePipeline create(boolean isBinary) {
      return isBinary ? binary() : multiClass();
   }

   /**
    * Creates a new DiscretePipeline for binary or multi-class problems
    *
    * @param isBinary      True - create a pipeline for binary problems, False create a pipeline for multi-class
    *                      problems.
    * @param preprocessors the preprocessors
    * @return the DiscretePipeline
    */
   public static DiscretePipeline create(boolean isBinary, Preprocessor... preprocessors) {
      return isBinary ? binary(preprocessors) : multiClass(preprocessors);
   }

   /**
    * Creates a new DiscretePipeline for binary or multi-class problems
    *
    * @param isBinary          True - create a pipeline for binary problems, False create a pipeline for multi-class
    *                          problems.
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    * @return the DiscretePipeline
    */
   public static DiscretePipeline create(boolean isBinary, DiscreteVectorizer featureVectorizer, Preprocessor... preprocessors) {
      return isBinary ? binary(featureVectorizer, preprocessors) : multiClass(featureVectorizer, preprocessors);
   }


   /**
    * Creates a new DiscretePipeline for unsupervised problems using a {@link NoOptVectorizer} as the label vectorizer.
    *
    * @param preprocessors the preprocessors
    * @return the DiscretePipeline
    */
   public static DiscretePipeline unsupervised(Preprocessor... preprocessors) {
      return unsupervised(new CountFeatureVectorizer(), preprocessors);
   }

   /**
    * Creates a new DiscretePipeline for unsupervised problems using a {@link NoOptVectorizer} as the label vectorizer.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    * @return the DiscretePipeline
    */
   public static DiscretePipeline unsupervised(DiscreteVectorizer featureVectorizer, Preprocessor... preprocessors) {
      DiscretePipeline pipeline = new DiscretePipeline(NoOptVectorizer.INSTANCE);
      pipeline.featureVectorizer = featureVectorizer;
      pipeline.preprocessorList.addAll(preprocessors);
      return pipeline;
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
