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
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.CountFeatureVectorizer;
import com.gengoai.apollo.ml.vectorizer.DiscreteVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;

import java.io.Serializable;
import java.util.function.Consumer;

/**
 * <p>A pipeline defines the preprocessing steps to be applied to {@link Example}s and the {@link Vectorizer}s
 * associated with converting the features and labels of examples into {@link NDArray}. Subclasses may define other
 * information that is common for a type machine learning model, e.g. methods for validation.</p>
 *
 * @param <LV> the label vectorizer type parameter
 * @param <T>  the subclass type parameter
 * @author David B. Bracewell
 */
public abstract class Pipeline<LV extends Vectorizer, T extends Pipeline<LV, ?>> implements Copyable<T>, Serializable {
   private static final long serialVersionUID = 1L;

   /**
    * The label vectorizer to use to create NDArray for labels
    */
   public final LV labelVectorizer;
   /**
    * The preprocessor list used to preprocess examples
    */
   public final PreprocessorList preprocessorList = new PreprocessorList();
   /**
    * The feature vectorizer to use to create NDArray for features
    */
   public DiscreteVectorizer featureVectorizer = new CountFeatureVectorizer();


   /**
    * Instantiates a new Pipeline.
    *
    * @param labelVectorizer the label vectorizer
    */
   protected Pipeline(LV labelVectorizer) {
      this.labelVectorizer = labelVectorizer;
   }

   /**
    * Fits the elements of the pipeline to the given dataset and then transforms the given dataset using the
    * preprocessors associated with the pipeline
    *
    * @param dataset the dataset to use for fitting the preprocessors and vectorizers
    * @return the dataset with preprocessing applied
    */
   public ExampleDataset fitAndPreprocess(ExampleDataset dataset) {
      ExampleDataset preprocessed = preprocessorList.fitAndTransform(dataset);
      labelVectorizer.fit(preprocessed);
      featureVectorizer.fit(preprocessed);

      return preprocessed;
   }

   @Override
   public T copy() {
      return Cast.as(Copyable.deepCopy(this));
   }

   /**
    * Updates the pipeline using the given consumer
    *
    * @param updater the updater
    * @return this pipeline updated
    */
   public T update(Consumer<? extends T> updater) {
      updater.accept(Cast.as(this));
      return Cast.as(this);
   }

}//END OF ModelParameters
